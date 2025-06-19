import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("/home/ec2-user/ecg_ptbxl_benchmarking/data/ptbxl/raw100.npy", allow_pickle=True)
one_data = data[0]  # Shape: (1000, 12)

# Define lead names (standard 12-lead ECG)
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Create a figure with 12 subplots
fig, axes = plt.subplots(12, 1, figsize=(12, 18), sharex=True)
fig.suptitle("12-lead ECG: First 10 seconds", fontsize=16)

# Plot each lead in its respective subplot
for i in range(12):
    axes[i].plot(one_data[:, i], color='black', linewidth=1)
    axes[i].set_ylabel(lead_names[i])
    axes[i].grid(True)
    if i < 11:
        axes[i].tick_params(labelbottom=False)  # Hide x-axis labels except bottom
    else:
        axes[i].set_xlabel("Time (samples at 100Hz)")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
plt.savefig("ecg_12_leads.png")
plt.close()
