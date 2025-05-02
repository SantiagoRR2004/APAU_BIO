import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

print(medmnist.__version__)
from medmnist import PneumoniaMNIST

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    print("ERROR: no GPU available")
    sys.exit(0)

currentDirectory = os.path.dirname(os.path.abspath(__file__))

# Dataset info
download = True
info = medmnist.INFO["pneumoniamnist"]
print(info)
DataClass = getattr(__import__("medmnist"), info["python_class"])

# Data transforms
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Load data
train_dataset = DataClass(split="train", transform=transform, download=download)
val_dataset = DataClass(split="val", transform=transform, download=download)
test_dataset = DataClass(split="test", transform=transform, download=download)

labels_train = [label for img, label in train_dataset]
labels_val = [label for img, label in val_dataset]
labels_test = [label for img, label in test_dataset]

print(f"Casos de nuemonia en train: {np.sum(labels_train)}")
print(f"En val: {np.sum(labels_val)}")
print(f"En test: {np.sum(labels_test)}")

##################################################################################################################
##################################################################################################################
##################################################################################################################
## Visualize some training images

healthy_images = []
pneumonia_images = []

for img, label in train_dataset:
    if label.item() == 0:
        healthy_images.append(img.squeeze())  # remove channel dimension
    if len(healthy_images) == 12:
        break

for img, label in train_dataset:
    if label.item() == 1:
        pneumonia_images.append(img.squeeze())  # remove channel dimension
    if len(pneumonia_images) == 12:
        break

# Create mosaic
fig, axes = plt.subplots(3, 4, figsize=(5, 4))
fig.suptitle("Healthy (Label = 0) Images from PneumoniaMNIST", fontsize=16)
for i, ax in enumerate(axes.flat):
    ax.imshow(healthy_images[i], cmap="gray")
    ax.axis("off")
plt.tight_layout()

# Create mosaic
fig, axes = plt.subplots(3, 4, figsize=(5, 4))
fig.suptitle("Non-healthy (Label = 1) Images from PneumoniaMNIST", fontsize=16)
for i, ax in enumerate(axes.flat):
    ax.imshow(pneumonia_images[i], cmap="gray")
    ax.axis("off")
plt.tight_layout()

##################################################################################################################
##################################################################################################################
##################################################################################################################


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (28x28x1)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding="same"),  # (28x28x16)
            torch.nn.MaxPool2d(kernel_size=2),  # (14x14x16)
            torch.nn.Conv2d(16, 8, kernel_size=3, padding="same"),  # (14x14x8)
            torch.nn.MaxPool2d(kernel_size=2),  # (7x7x8)
            torch.nn.Conv2d(8, 8, kernel_size=3, padding="same"),  # (7x7x8)
            torch.nn.MaxPool2d(kernel_size=2),  # (4x4x8)
            torch.nn.Conv2d(8, 8, kernel_size=3, padding="same"),  # (4x4x8)
        )

        # Decoder (4x4x8)
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),  # (8x8x8)
            torch.nn.ConvTranspose2d(8, 8, kernel_size=3, padding=(1, 1)),  # (8x8x8)
            torch.nn.Upsample(scale_factor=2),  # (16x16x8)
            torch.nn.ConvTranspose2d(8, 16, kernel_size=3),  # (14x14x16)
            torch.nn.Upsample(scale_factor=2),  # (28x28x16)
            torch.nn.ConvTranspose2d(16, 1, kernel_size=3, padding=(1, 1)),  # (28x28x1)
            torch.nn.Sigmoid(),  # Necessary to have the same range as the input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


modelAE = AE().to(device)

##################################################################################################################
##################################################################################################################
##################################################################################################################

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(modelAE.parameters())
num_epochs = 25
batch_size = 60

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)


##################################################################################################################
##################################################################################################################
##################################################################################################################
# --------------------------------
# AE training
# --------------------------------


loss_v = np.empty(0)
loss_val_v = np.empty(0)

for epoch in range(num_epochs):
    modelAE.train()
    total_loss = 0.0
    total_loss_val = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = modelAE(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # # statistics after a batch
        total_loss += loss.item()

    modelAE.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val = modelAE(inputs_val)
            loss_val = criterion(outputs_val, inputs_val)
            total_loss_val += loss_val.item()

    average_loss = total_loss / len(train_loader)
    average_loss_val = total_loss_val / len(val_loader)
    loss_v = np.append(loss_v, average_loss)
    loss_val_v = np.append(loss_val_v, average_loss_val)

    print(
        "Epoch {:02d}: loss {:.4f} - val. loss {:.4f}".format(
            epoch + 1, average_loss, average_loss_val
        )
    )

torch.save(modelAE.state_dict(), os.path.join(currentDirectory, "PneumoniaAE.pth"))


##################################################################################################################
##################################################################################################################
##################################################################################################################


plt.show()
