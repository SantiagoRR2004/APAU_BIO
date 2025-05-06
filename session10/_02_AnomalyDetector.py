import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))

custom_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
)

mnist_dataset = datasets.MNIST(
    root="/tmp/data", download=True, transform=custom_transform
)
print("Length of dataset:", len(mnist_dataset))


val_data = Subset(mnist_dataset, list(range(50000, len(mnist_dataset))))
print("Length of validation dataset: ", len(val_data))

val_loader = DataLoader(val_data, len(val_data), shuffle=False, num_workers=2)


# --------------------------------
# AE definition
# --------------------------------


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder 784 -> 256 -> 48 -> 2
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 2),
        )

        # Decoder 2 -> 48 -> 256 -> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AE()
model.load_state_dict(
    torch.load(os.path.join(currentDirectory, "_01_AE.pth"), weights_only=True)
)
model.eval()

for batch_data, batch_labels in val_loader:
    inputs_val = batch_data  # Just one batch
    inputs_labels = batch_labels
outputs_val = model(inputs_val)

mse = torch.mean((inputs_val - outputs_val) ** 2, dim=1)

# ---------------------------------------------
# Best reconstructed digits
# ---------------------------------------------

sorted_idx = np.argsort(mse.detach())

best_idx = np.zeros(10, dtype=int)
for i in range(10):
    idx = 0
    while batch_labels[sorted_idx[idx]] != i:
        idx += 1
    best_idx[i] = sorted_idx[idx]

input_images = inputs_val[best_idx]
input_labels = inputs_labels[best_idx].numpy()

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(10):
    img = input_images[i].reshape((28, 28))
    ax = fig.add_subplot(1, 10, i + 1)
    ax.axis("off")
    ax.set_title(input_labels[i])
    fig.suptitle("Original digits")
    ax.imshow(img, cmap="binary")
fig.tight_layout()
fig.savefig(os.path.join(currentDirectory, "_02_BestReconstr_input.png"))


fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(10):
    img = outputs_val[best_idx[i]].reshape((28, 28)).detach()
    ax = fig.add_subplot(1, 10, i + 1)
    ax.axis("off")
    ax.set_title(input_labels[i])
    fig.suptitle("Best reconstructed digits (undercomplete AE)")
    ax.imshow(img, cmap="binary")
fig.tight_layout()
fig.savefig(os.path.join(currentDirectory, "_02_BestReconstr_output.png"))

# ---------------------------------------------
# Worst reconstructed digits
# ---------------------------------------------

n_to_show = 20

worst_idx = sorted_idx[-n_to_show:]

input_images = inputs_val[worst_idx]
input_labels = inputs_labels[worst_idx].numpy()

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    img = input_images[i].reshape((28, 28))
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.set_title(input_labels[i])
    fig.suptitle("Original digits")
    ax.imshow(img, cmap="binary")
fig.tight_layout()
fig.savefig(os.path.join(currentDirectory, "_02_WorstReconstr_input.png"))

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(n_to_show):
    img = outputs_val[worst_idx[i]].reshape((28, 28)).detach()
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.set_title(input_labels[i])
    fig.suptitle("Worst reconstructed digits (undercomplete AE)")
    ax.imshow(img, cmap="binary")
fig.tight_layout()
fig.savefig(os.path.join(currentDirectory, "_02_WorstReconstr_output.png"))
