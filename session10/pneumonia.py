import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

print(medmnist.__version__)
from medmnist import PneumoniaMNIST


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

plt.show()
