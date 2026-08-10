import matplotlib.pyplot as plt

epochs = range(30)
train_loss = [0.444863, 0.250142, 0.226046, 0.211317, 0.216650, 0.214392, 0.214601, 0.216742, 0.222722, 0.211663, 
              0.212735, 0.212165, 0.210118, 0.204492, 0.206302, 0.204411, 0.202305, 0.200056, 0.197953, 0.194577, 
              0.191167, 0.190977, 0.189279, 0.189436, 0.184286, 0.178027, 0.183185, 0.176564, 0.181921, 0.178493]
valid_loss = [0.352898, 0.289525, 0.276707, 0.311945, 0.282973, 0.280345, 0.283135, 0.280979, 0.289775, 0.300537, 
              0.299602, 0.305624, 0.283004, 0.299328, 0.285938, 0.287908, 0.288925, 0.293531, 0.285344, 0.289656, 
              0.301933, 0.317811, 0.298850, 0.303210, 0.306397, 0.303114, 0.311969, 0.305512, 0.308477, 0.309375]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, valid_loss, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
