from utils import utils
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import torch.nn as nn
import torch.nn.functional as F
sampling_frequency=100
datafolder='/home/ec2-user/ecg_ptbxl_benchmarking/data/ptbxl/'
task='all'
outputfolder='/home/ec2-user/ecg_ptbxl_benchmarking/output/'
import torch
import numpy as np
from collections import Counter

# Load PTB-XL data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
# Preprocess label data
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
# Select relevant data and convert to one-hot
data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

# 1-9 for training 
X_train = data[labels.strat_fold < 10]
y_train = Y[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]

num_classes = 71         # <=== number of classes in the finetuning dataset
input_shape = [1000,12] # <=== shape of samples, [None, 12] in case of different lengths

print("Training and validation data shapes:")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


from models.fastai_model import fastai_model

experiment = 'exp0'
modelname = 'fastai_xresnet1d101'
pretrainedfolder = '/home/ec2-user/ecg_ptbxl_benchmarking/output/exp0/models/fastai_xresnet1d101/'
mpath='/home/ec2-user/ecg_ptbxl_benchmarking/output/' # <=== path where the finetuned model will be stored
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


import pickle

standard_scaler = pickle.load(open('/home/ec2-user/ecg_ptbxl_benchmarking/output/exp0/data/standard_scaler.pkl', "rb"))

X_train = utils.apply_standardizer(X_train, standard_scaler)
X_val = utils.apply_standardizer(X_val, standard_scaler)

# Predicting
# y_val_pred = model.predict(X_val) 
# results = utils.evaluate_experiment(y_val, y_val_pred)
# print("Evaluation results:", results)

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

        # α_ij^k coefficients (1D: just along L)
        sum_A = A.sum(dim=2, keepdim=True)  # (B, C, 1)
        numerator = grads2
        denom = 2 * grads2 + sum_A * grads3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom) * eps)
        alphas = numerator / denom

        alphas = alphas.clamp(min=0)
        weights = (alphas * grads3).sum(dim=2, keepdim=True)  # (B, C, 1)

        cam = F.relu(torch.sum(weights * A, dim=1, keepdim=True))  # (B,1,L)

        cam = F.interpolate(cam, size=L, mode='linear', align_corners=False)
        cam = cam.squeeze(1)

        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + eps)

        return cam  # shape (B, L)

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



def colorline(time, signal, heatmap, cmap='rainbow'):
    points = np.array([time, signal]).T.reshape(-1,1,2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segs, array=heatmap, cmap=cmap)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def plot_signal_cam(signal, cam, time=None, title=None):
    L = len(signal)
    if time is None:
        time = np.linspace(0, 1, L)
    plt.figure()
    lc = colorline(time, signal, cam)
    plt.colorbar(lc, label='Grad‑CAM++')
    plt.title(title or "Signal w/ CAM")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("temp.png")

def find_last_conv1d(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv1d):
            return layer
    raise ValueError("No Conv1d layer found.")

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

# 2. Pick sample dataset
datas = X_val[:10]  # Use first 10 samples for visualization

# 3. Compute dataset-level importance
# labels, percents = dataset_cam_stats(pytorch_model, gradcampp, datas)
# print("Top activation positions:", list(zip(labels, percents)))

# 4. Plot one example with CAM overlay
sample = datas[0]
# If sample is shape (1000, 12) → transpose to (12, 1000)
sample = sample.squeeze().T
print(sample.shape)  # Should be (12, 1000)

# Now add batch dimension: [1, 12, 1000]
x_tensor = torch.tensor(sample[None, ...], dtype=torch.float32).to(device)

# Generate CAM
cam0 = gradcampp.generate(x_tensor)[0].cpu().numpy()
plot_signal_cam(sample, cam0)
