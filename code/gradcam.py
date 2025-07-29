from utils import utils
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import Counter
from matplotlib import cm
from matplotlib.colors import Normalize
import pickle
from models.fastai_model import fastai_model
import json
import os

class GradCAMPlusPlus1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Attach hooks
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # input_tensor: shape (B, C, L)
        self.model.zero_grad()
        outputs = self.model(input_tensor)  # (B, num_classes)

        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
        one_hot = torch.zeros_like(outputs)
        one_hot[range(outputs.size(0)), class_idx] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)

        A = self.activations      # (B, C, L)
        grads = self.gradients    # (B, C, L)

        grads2 = grads ** 2
        grads3 = grads ** 3
        eps = 1e-8
        B, C, L = grads.shape

        # Grad-CAM++ α_ij^k weights
        sum_A = A.sum(dim=2, keepdim=True)  # (B, C, 1)
        numerator = grads2
        denom = 2 * grads2 + sum_A * grads3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom) * eps)
        alphas = numerator / denom
        alphas = alphas.clamp(min=0)

        weights = (alphas * grads3).sum(dim=2, keepdim=True)  # (B, C, 1)

        # 💡 Preserve channel dimension in output
        cam = F.relu(weights * A)  # (B, C, L)

        # (Optional) Interpolate if CAM has smaller L than input
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear', align_corners=False)

        # Normalize CAM per channel
        cam_min = cam.amin(dim=2, keepdim=True)
        cam_max = cam.amax(dim=2, keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + eps)  # (B, C, L)

        return cam  # shape: (B, C, L)

def dataset_cam_stats(model, gradcampp, datas, topk=1):
    cams = []
    for x in datas:
        with torch.no_grad():
            x_tensor = torch.tensor(x[None,...], dtype=torch.float).transpose(1, 2).to(model.device)
        cam = gradcampp.generate(x_tensor)[0].cpu().numpy()
        cams.append(cam)
    # Find most activated index per signal
    peak_idxs = [cam.argmax() for cam in cams]
    freq = Counter(peak_idxs)
    # Convert to percentages
    total = len(cams)
    labels, counts = zip(*freq.items())
    percents = [100 * c / total for c in counts]
    return labels, percents

# Apply softmax to convert to probabilities that sum to 1
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max to prevent numerical instability
    return e_x / e_x.sum(axis=0)


def colorline(time, signal, heatmap, cmap='rainbow'):
    points = np.array([time, signal]).T.reshape(-1,1,2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segs, array=heatmap, cmap=cmap)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def plot_signal_cam_ecg_style(sample, cam, fs=100, lead_names=None, save_path=None):
    """
    Plot ECG signal with Grad-CAM++ overlay in standard 12-lead layout.

    sample: (12, T) ECG signal
    cam:    (12, T) Grad-CAM++ values
    fs:     sampling frequency (default=100)
    lead_names: list of lead names
    save_path: if provided, saves figure instead of showing it
    """
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    n_channels, n_timesteps = sample.shape
    time = np.linspace(0, n_timesteps / fs, n_timesteps)

    # Normalize CAM per lead
    cam_norm = (cam - cam.min(axis=1, keepdims=True)) / (cam.max(axis=1, keepdims=True) - cam.min(axis=1, keepdims=True) + 1e-6)

    fig, axes = plt.subplots(12, 1, figsize=(13, 20), sharex=True,
                             gridspec_kw={'height_ratios': [1]*12, 'hspace': 0.3})
    fig.suptitle("12-lead ECG with Grad-CAM++ Overlay", fontsize=18)

    # Shared colormap and normalization for colorbar
    cmap = cm.get_cmap('jet')
    norm = Normalize(vmin=0, vmax=1)

    for i in range(n_channels):
        ax = axes[i]
        ax.plot(time, sample[i], color='black', linewidth=1)
        ax.imshow(cam_norm[i][None, :], aspect='auto',
                  extent=[time[0], time[-1], sample[i].min(), sample[i].max()],
                  cmap=cmap, norm=norm, alpha=0.4, interpolation='bilinear', origin='lower')
        ax.set_ylabel(lead_names[i])
        ax.grid(True)

        if i < n_channels - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (s)")

    # Add a colorbar to the right of the subplots
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', pad=0.01)
    cbar.set_label("Feature Importance (Grad-CAM++)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.98, 0.96])  # Adjust for title and colorbar

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_only_signal(one_data, save_path=None):
    """
    Plot standard 12-lead ECG without Grad-CAM overlay.

    one_data: (12, T) ECG signal
    save_path: if provided, saves figure instead of showing it
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(12, 1, figsize=(12, 18), sharex=True)
    fig.suptitle("12-lead ECG: First 10 seconds", fontsize=16)

    for i in range(12):
        axes[i].plot(one_data[i], color='black', linewidth=1)
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True)
        if i < 11:
            axes[i].tick_params(labelbottom=False)
        else:
            axes[i].set_xlabel("Time (samples at 100Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



def find_last_conv1d(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv1d):
            return layer
    raise ValueError("No Conv1d layer found.")

sampling_frequency=100
datafolder='/home/ec2-user/ecg_ptbxl_benchmarking/data/ptbxl/'
task='all'
outputfolder='/home/ec2-user/ecg_ptbxl_benchmarking/output/'

if __name__ == "__main__":
    # Create output directories
    categories = {
        "mi_correct": "MI_related_correct_predictions",
        "mi_incorrect": "MI_related_incorrect_predictions",
        "non_mi_correct": "Non_MI_correct_predictions",
        "non_mi_incorrect": "Non_MI_incorrect_predictions"
    }

    # Create main output directory and subdirectories
    main_output_dir = os.path.join(outputfolder, "gradcam_categories")
    os.makedirs(main_output_dir, exist_ok=True)
    for dir_name in categories.values():
        os.makedirs(os.path.join(main_output_dir, dir_name), exist_ok=True)

    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)
    
    # Validation set (fold 10)
    val_mask = labels.strat_fold == 10
    X_val = data[val_mask]
    y_val = Y[val_mask]
    raw_val_labels = raw_labels[val_mask]  # Get raw labels for MI classification

    # Load SCP mapping
    with open('/home/ec2-user/ecg_ptbxl_benchmarking/index_to_scp.json', 'r') as f:
        index_to_scp = json.load(f)
    scp_to_index = {v: k for k, v in index_to_scp.items()}  # Reverse mapping
    
    MI_related_scps = ["IMI", "ASMI", "ILMI", "ALMI", "LMI", "PMI", "IPMI", "IPLMI"]
    MI_related_indices = [scp_to_index[scp] for scp in MI_related_scps if scp in scp_to_index]

    # Model configuration
    num_classes = 71
    input_shape = [1000, 12]
    experiment = 'exp0'
    modelname = 'fastai_xresnet1d101'
    pretrainedfolder = '/home/ec2-user/ecg_ptbxl_benchmarking/output/exp0/models/fastai_xresnet1d101/'
    mpath = '/home/ec2-user/ecg_ptbxl_benchmarking/output/'
    n_classes_pretrained = 71 # <=== because we load the model from exp0, this should be fixed because this depends the experiment

    model = fastai_model(
        modelname, 
        num_classes, 
        sampling_frequency, 
        mpath, 
        input_shape=input_shape, 
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained, 
        pretrained=True,
        epochs_finetuning=0,
    )
    # Load and apply standard scaler
    standard_scaler = pickle.load(open('/home/ec2-user/ecg_ptbxl_benchmarking/output/exp0/data/standard_scaler.pkl', "rb"))
    X_val = utils.apply_standardizer(X_val, standard_scaler)

    # Get predictions
    y_val_pred = model.predict(X_val)
    results = utils.evaluate_experiment(y_val, y_val_pred)
    print("Evaluation results:", results)

    # Grad-CAM visualization
    pytorch_model = model.get_model(X_val[:10])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model.to(device)

    for i, layer in enumerate(pytorch_model):
        print(f"Layer {i}: {layer.__class__.__name__}")
    target_layer = find_last_conv1d(pytorch_model)
    print("Target layer for Grad-CAM++:", target_layer)
    # 1. Initialize model & apply Grad‑CAM++ hook]
    gradcampp = GradCAMPlusPlus1D(pytorch_model, target_layer)

    # Processing parameters
    batch_size = 100  # Adjust based on memory constraints
    num_samples = len(X_val)
    processed = 0

    # Tracking counters
    counters = {cat: 0 for cat in categories.keys()}
    target_count = 50
    # Continue until we have enough samples in all categories
    while not all(count >= target_count for count in counters.values()) and processed < num_samples:
        # Calculate batch boundaries
        batch_start = processed
        batch_end = min(processed + batch_size, num_samples)
        batch_X = X_val[batch_start:batch_end]
        batch_y_true = y_val[batch_start:batch_end]
        batch_raw_labels = raw_val_labels.iloc[batch_start:batch_end]
        batch_y_pred = y_val_pred[batch_start:batch_end]
        
        # Process each sample in the batch
        for i in range(len(batch_X)):
            # Skip if we already have enough samples in all categories
            if all(count >= target_count for count in counters.values()):
                break
                
            sample_idx = batch_start + i
            x = batch_X[i]
            y_true = batch_y_true[i]
            y_pred = batch_y_pred[i]
            print(y_pred)
            print(np.sum(y_pred))

            # Get prediction confidence
            y_pred_softmax = softmax(y_pred)
            
            # Calculate total confidence for MI-related predictions
            # print(f"MI-related indices: {MI_related_indices}")
            mi_confidence = sum(y_pred_softmax[int(idx)] for idx in MI_related_indices)

            # Convert to percentage
            mi_confidence_percent = mi_confidence * 100

            print(f"MI-related confidence (after softmax): {mi_confidence_percent:.2f}%")
            
            # Determine true label SCPs
            true_scp_indices = np.where(y_true == 1)[0]
            true_scps = [index_to_scp[str(idx)] for idx in true_scp_indices if str(idx) in index_to_scp]
            
            # Classify as MI-related or not
            is_mi = any(scp in MI_related_scps for scp in true_scps)
            
            # Determine prediction correctness (using argmax for simplicity)
            # Adjust this if you need a different threshold for positive prediction
            pred_class = np.argmax(y_pred)
            is_correct = pred_class in true_scp_indices
            
            # Determine category
            if is_mi:
                category = "mi_correct" if is_correct else "mi_incorrect"
            else:
                category = "non_mi_correct" if is_correct else "non_mi_incorrect"
            
            # Skip if we already have enough in this category
            if counters[category] >= target_count:
                continue
            
            # Generate Grad-CAM
            try:
                # Preprocess sample
                s_processed = x.squeeze().T  # (12, 1000)
                x_tensor = torch.tensor(s_processed[None, ...], dtype=torch.float32).to(device)
                
                # Generate CAM
                cam = gradcampp.generate(x_tensor)[0].cpu().numpy()  # (1000,)
                
                # Create filename with sample info
                filename_base = f"sample_{sample_idx}_mi_{is_mi}_correct_{is_correct}"
                
                # Save plots
                output_dir = os.path.join(main_output_dir, categories[category])
                
                # Save ECG only
                plot_only_signal(
                    s_processed, 
                    save_path=os.path.join(output_dir, f"{filename_base}_{mi_confidence_percent}_signal.png")
                )
                
                # Save ECG + CAM overlay
                plot_signal_cam_ecg_style(
                    sample=s_processed, 
                    cam=cam, 
                    fs=sampling_frequency, 
                    save_path=os.path.join(output_dir, f"{filename_base}_{mi_confidence_percent}_gradcam.png")
                )
                
                # Increment counter
                counters[category] += 1
                print(f"Saved {category} #{counters[category]}/{target_count} (Sample {sample_idx})")
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {str(e)}")
                continue
        
        # Update processed count
        processed = batch_end
        print(f"Processed {processed}/{num_samples} samples. Current counts: {counters}")
    
    # Final status
    print("Processing complete. Final counts:")
    for cat, count in counters.items():
        print(f"{categories[cat]}: {count}/{target_count}")