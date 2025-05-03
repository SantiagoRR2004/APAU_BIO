import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import random
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
## Change datasets

# Training can only have healthy images
train_dataset = [img for img in train_dataset if img[1] == 0]
print(f"Casos sin neumonía en train: {len(train_dataset)}")

# We divide validation and test sets into healthy and pneumonia
val_HealthyDataset = [img for img in val_dataset if img[1] == 0]
val_PneumoniaDataset = [img for img in val_dataset if img[1] == 1]
test_HealthyDataset = [img for img in test_dataset if img[1] == 0]
test_PneumoniaDataset = [img for img in test_dataset if img[1] == 1]

# We balance the datasets
minVal = min(len(val_HealthyDataset), len(val_PneumoniaDataset))
minTest = min(len(test_HealthyDataset), len(test_PneumoniaDataset))
val_HealthyDataset = random.sample(val_HealthyDataset, minVal)
val_PneumoniaDataset = random.sample(val_PneumoniaDataset, minVal)
test_HealthyDataset = random.sample(test_HealthyDataset, minTest)
test_PneumoniaDataset = random.sample(test_PneumoniaDataset, minTest)

print(f"Casos sin neumonía en val: {len(val_HealthyDataset)}")
print(f"Casos con neumonía en val: {len(val_PneumoniaDataset)}")
print(f"Casos sin neumonía en test: {len(test_HealthyDataset)}")
print(f"Casos con neumonía en test: {len(test_PneumoniaDataset)}")

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
val_loader_healty = DataLoader(
    val_HealthyDataset, batch_size, shuffle=False, num_workers=2
)
val_loader_pneumonia = DataLoader(
    val_PneumoniaDataset, batch_size, shuffle=False, num_workers=2
)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

##################################################################################################################
##################################################################################################################
##################################################################################################################
# --------------------------------
# AE training
# --------------------------------


loss_v = np.empty(0)
loss_val_v = np.empty(0)
val_healthy_distances = np.empty(0)
val_pneumonia_distances = np.empty(0)

for epoch in range(num_epochs):
    modelAE.train()
    total_loss = 0.0
    total_loss_val_healthy = 0.0
    total_loss_val_anomaly = 0.0
    total_difference = 0.0
    total_difference_val_healthy = 0.0
    total_difference_val_anomaly = 0.0
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
        total_difference += torch.sum(torch.abs(outputs - inputs)).item()

    modelAE.eval()
    with torch.no_grad():
        # Evaluamos con los de validación sanos para encontrar el umbral
        for _, data in enumerate(val_loader_healty, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val = modelAE(inputs_val)
            loss_val = criterion(outputs_val, inputs_val)

            # Statistics for healthy validation
            total_loss_val_healthy += loss_val.item()
            total_difference_val_healthy += torch.sum(
                torch.abs(outputs_val - inputs_val)
            ).item()
            # Distances for the threshold
            if epoch == num_epochs - 1:
                val_healthy_distances = np.append(
                    val_healthy_distances,
                    torch.sum(torch.abs(outputs_val - inputs_val), dim=(1, 2, 3))
                    .cpu()
                    .numpy(),
                )

        # Evaluamos con los de validación anómalos para encontrar el umbral
        for _, data in enumerate(val_loader_pneumonia, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val = modelAE(inputs_val)
            loss_val = criterion(outputs_val, inputs_val)

            # Statistics for pneumonia validation
            total_loss_val_anomaly += loss_val.item()
            total_difference_val_anomaly += torch.sum(
                torch.abs(outputs_val - inputs_val)
            ).item()
            # Distances for the threshold
            if epoch == num_epochs - 1:
                val_pneumonia_distances = np.append(
                    val_pneumonia_distances,
                    torch.sum(torch.abs(outputs_val - inputs_val), dim=(1, 2, 3))
                    .cpu()
                    .numpy(),
                )

    # Calculate the loss by averaging over the batches
    average_loss = total_loss / len(train_loader)
    average_loss_val_healthy = total_loss_val_healthy / len(val_loader_healty)
    average_loss_val_anomaly = total_loss_val_anomaly / len(val_loader_pneumonia)

    # Calculate the difference by averaging over each image
    average_difference = total_difference / len(train_dataset)
    average_difference_val_healthy = total_difference_val_healthy / len(
        val_HealthyDataset
    )
    average_difference_val_anomaly = total_difference_val_anomaly / len(
        val_PneumoniaDataset
    )

    # Store in the lists
    loss_v = np.append(loss_v, average_loss)
    loss_val_v = np.append(loss_val_v, average_loss_val_healthy)

    print(
        "Epoch {:02d}: loss {:.4f} - val. healthy loss {:.4f} - val. pneumonia loss {:.4f} - MAE {:.4f} - val. healthy MAE {:.4f} - val. pneumonia MAE {:.4f}".format(
            epoch + 1,
            average_loss,
            average_loss_val_healthy,
            average_loss_val_anomaly,
            average_difference,
            average_difference_val_healthy,
            average_difference_val_anomaly,
        )
    )

labels = [0] * len(val_healthy_distances) + [1] * len(val_pneumonia_distances)
scores = np.concatenate((val_healthy_distances, val_pneumonia_distances), axis=0)

fpr, tpr, thresholds = roc_curve(labels, scores)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
print("Best threshold:", best_threshold)

torch.save(modelAE.state_dict(), os.path.join(currentDirectory, "PneumoniaAE.pth"))


##################################################################################################################
##################################################################################################################
##################################################################################################################
# Plotting the ROC Curve
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.scatter(
    fpr[np.argmax(tpr - fpr)],
    tpr[np.argmax(tpr - fpr)],
    marker="o",
    color="red",
    label="Best Threshold",
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid()

##################################################################################################################
##################################################################################################################
##################################################################################################################


plt.show()
