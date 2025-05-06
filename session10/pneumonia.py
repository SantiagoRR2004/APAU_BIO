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


# Use the MNIST digits
from _01_UndercompleteAE import AE

modelAE = AE().to(device)

##################################################################################################################
##################################################################################################################
##################################################################################################################


class VAE(torch.nn.Module):
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

        self.mean_layer = torch.nn.Conv2d(
            8, 8, kernel_size=3, padding="same"
        )  # (4x4x8)
        self.logvar_layer = torch.nn.Conv2d(
            8, 8, kernel_size=3, padding="same"
        )  # (4x4x8)

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

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


# Use the MNIST digits
from _04_VAE import VAE

modelVAE = VAE().to(device)

##################################################################################################################
##################################################################################################################
##################################################################################################################

criterion = torch.nn.BCELoss()
optimizerAE = torch.optim.Adam(modelAE.parameters())
optimizerVAE = torch.optim.Adam(modelVAE.parameters())
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
# Lists to store for plotting the ae
ae_loss_v = np.empty(0)
ae_loss_val_health_v = np.empty(0)
ae_loss_val_anomaly_v = np.empty(0)
ae_distances = np.empty(0)
ae_val_healthy_distances = np.empty(0)
ae_val_pneumonia_distances = np.empty(0)
ae_val_healthy_distances_Threshold = np.empty(0)
ae_val_pneumonia_distances_Threshold = np.empty(0)

# Lists to store for plotting the vae
vae_loss_v = np.empty(0)
vae_loss_val_health_v = np.empty(0)
vae_loss_val_anomaly_v = np.empty(0)
vae_distances = np.empty(0)
vae_val_healthy_distances = np.empty(0)
vae_val_pneumonia_distances = np.empty(0)
vae_val_healthy_distances_Threshold = np.empty(0)
vae_val_pneumonia_distances_Threshold = np.empty(0)

# Epochs for plotting
epochs = range(1, num_epochs + 1)

# Labels for the roc
labels = [0] * len(val_HealthyDataset) + [1] * len(val_PneumoniaDataset)

##################################################################################################################
##################################################################################################################
##################################################################################################################
# --------------------------------
# AE training
# --------------------------------


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
        optimizerAE.zero_grad()
        outputs = modelAE(inputs).view(-1, 1, 28, 28)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizerAE.step()

        # # statistics after a batch
        total_loss += loss.item()
        total_difference += torch.sum(torch.abs(outputs - inputs)).item()

    modelAE.eval()
    with torch.no_grad():
        # Evaluamos con los de validación sanos para encontrar el umbral
        for _, data in enumerate(val_loader_healty, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val = modelAE(inputs_val).view(-1, 1, 28, 28)
            loss_val = criterion(outputs_val, inputs_val)

            # Statistics for healthy validation
            total_loss_val_healthy += loss_val.item()
            total_difference_val_healthy += torch.sum(
                torch.abs(outputs_val - inputs_val)
            ).item()
            # Distances for the threshold
            if epoch == num_epochs - 1:
                ae_val_healthy_distances_Threshold = np.append(
                    ae_val_healthy_distances_Threshold,
                    torch.sum(torch.abs(outputs_val - inputs_val), dim=(1, 2, 3))
                    .cpu()
                    .numpy(),
                )

        # Evaluamos con los de validación anómalos para encontrar el umbral
        for _, data in enumerate(val_loader_pneumonia, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val = modelAE(inputs_val).view(-1, 1, 28, 28)
            loss_val = criterion(outputs_val, inputs_val)

            # Statistics for pneumonia validation
            total_loss_val_anomaly += loss_val.item()
            total_difference_val_anomaly += torch.sum(
                torch.abs(outputs_val - inputs_val)
            ).item()
            # Distances for the threshold
            if epoch == num_epochs - 1:
                ae_val_pneumonia_distances_Threshold = np.append(
                    ae_val_pneumonia_distances_Threshold,
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
    ae_loss_v = np.append(ae_loss_v, average_loss)
    ae_loss_val_health_v = np.append(ae_loss_val_health_v, average_loss_val_healthy)
    ae_loss_val_anomaly_v = np.append(ae_loss_val_anomaly_v, average_loss_val_anomaly)
    ae_distances = np.append(ae_distances, average_difference)
    ae_val_healthy_distances = np.append(
        ae_val_healthy_distances, average_difference_val_healthy
    )
    ae_val_pneumonia_distances = np.append(
        ae_val_pneumonia_distances, average_difference_val_anomaly
    )

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

torch.save(modelAE.state_dict(), os.path.join(currentDirectory, "PneumoniaAE.pth"))

##################################################################################################################
##################################################################################################################
##################################################################################################################
# --------------------------------
# VAE training
# --------------------------------


def loss_function_vae(recon_x, x, mu, log_var):
    recon_loss = torch.nn.BCELoss()(recon_x, x)
    # print(recon_loss)
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    # print(recon_loss)
    # print(kl_loss)
    # sys.exit(0)
    # Total loss: reconstruction + KL regularization
    return recon_loss + 0.01 * kl_loss


for epoch in range(num_epochs):
    modelVAE.train()
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
        optimizerVAE.zero_grad()
        outputs, mean, logvar = modelVAE(inputs)
        outputs = outputs.view(-1, 1, 28, 28)
        loss = loss_function_vae(outputs, inputs, mean, logvar)
        loss.backward()
        optimizerVAE.step()

        # # statistics after a batch
        total_loss += loss.item()
        total_difference += torch.sum(torch.abs(outputs - inputs)).item()

    modelVAE.eval()
    with torch.no_grad():
        # Evaluamos con los de validación sanos para encontrar el umbral
        for _, data in enumerate(val_loader_healty, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val, mean, logvar = modelVAE(inputs_val)
            outputs_val = outputs_val.view(-1, 1, 28, 28)
            loss_val = loss_function_vae(outputs_val, inputs_val, mean, logvar)

            # Statistics for healthy validation
            total_loss_val_healthy += loss_val.item()
            total_difference_val_healthy += torch.sum(
                torch.abs(outputs_val - inputs_val)
            ).item()
            # Distances for the threshold
            if epoch == num_epochs - 1:
                vae_val_healthy_distances_Threshold = np.append(
                    vae_val_healthy_distances_Threshold,
                    torch.sum(torch.abs(outputs_val - inputs_val), dim=(1, 2, 3))
                    .cpu()
                    .numpy(),
                )

        # Evaluamos con los de validación anómalos para encontrar el umbral
        for _, data in enumerate(val_loader_pneumonia, 0):
            inputs_val, _ = data
            inputs_val = inputs_val.to(device)
            outputs_val, mean, logvar = modelVAE(inputs_val)
            outputs_val = outputs_val.view(-1, 1, 28, 28)
            loss_val = loss_function_vae(outputs_val, inputs_val, mean, logvar)

            # Statistics for pneumonia validation
            total_loss_val_anomaly += loss_val.item()
            total_difference_val_anomaly += torch.sum(
                torch.abs(outputs_val - inputs_val)
            ).item()
            # Distances for the threshold
            if epoch == num_epochs - 1:
                vae_val_pneumonia_distances_Threshold = np.append(
                    vae_val_pneumonia_distances_Threshold,
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
    vae_loss_v = np.append(vae_loss_v, average_loss)
    vae_loss_val_health_v = np.append(vae_loss_val_health_v, average_loss_val_healthy)
    vae_loss_val_anomaly_v = np.append(vae_loss_val_anomaly_v, average_loss_val_anomaly)
    vae_distances = np.append(vae_distances, average_difference)
    vae_val_healthy_distances = np.append(
        vae_val_healthy_distances, average_difference_val_healthy
    )
    vae_val_pneumonia_distances = np.append(
        vae_val_pneumonia_distances, average_difference_val_anomaly
    )

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

torch.save(modelVAE.state_dict(), os.path.join(currentDirectory, "PneumoniaVAE.pth"))


##################################################################################################################
##################################################################################################################
##################################################################################################################
# Calculate the roc curves

scores_ae = np.concatenate(
    (ae_val_healthy_distances_Threshold, ae_val_pneumonia_distances_Threshold), axis=0
)
fpr_ae, tpr_ae, thresholds_ae = roc_curve(labels, scores_ae)
j_scores_ae = tpr_ae - fpr_ae
best_threshold_ae = thresholds_ae[np.argmax(j_scores_ae)]
print("Best threshold:", best_threshold_ae)

scores_vae = np.concatenate(
    (vae_val_healthy_distances_Threshold, vae_val_pneumonia_distances_Threshold), axis=0
)
fpr_vae, tpr_vae, thresholds_vae = roc_curve(labels, scores_vae)
j_scores_vae = tpr_vae - fpr_vae
best_threshold_vae = thresholds_vae[np.argmax(j_scores_vae)]
print("Best threshold:", best_threshold_vae)


##################################################################################################################
##################################################################################################################
##################################################################################################################
# Plotting the ROC Curve
roc_auc_ae = auc(fpr_ae, tpr_ae)
roc_auc_vae = auc(fpr_vae, tpr_vae)
plt.figure()
plt.plot(
    fpr_ae, tpr_ae, color="blue", lw=2, label=f"AE ROC curve (AUC = {roc_auc_ae:.2f})"
)
plt.plot(
    fpr_vae,
    tpr_vae,
    color="orange",
    lw=2,
    label=f"VAE ROC curve (AUC = {roc_auc_vae:.2f})",
)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.scatter(
    fpr_ae[np.argmax(tpr_ae - fpr_ae)],
    tpr_ae[np.argmax(tpr_ae - fpr_ae)],
    marker="o",
    color="red",
    label="Best Threshold AE",
)
plt.scatter(
    fpr_vae[np.argmax(tpr_vae - fpr_vae)],
    tpr_vae[np.argmax(tpr_vae - fpr_vae)],
    marker="s",
    color="green",
    label="Best Threshold VAE",
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
# Plot the losses
plt.figure()
plt.plot(epochs, ae_loss_v, "b-o", label="Training AE")
plt.plot(epochs, vae_loss_v, "b--s", label="Training VAE")
plt.plot(epochs, ae_loss_val_health_v, "g-o", label="Validation Healthy AE")
plt.plot(epochs, vae_loss_val_health_v, "g--s", label="Validation Healthy VAE")
plt.plot(epochs, ae_loss_val_anomaly_v, "r-o", label="Validation Pneumonia AE")
plt.plot(epochs, vae_loss_val_anomaly_v, "r--s", label="Validation Pneumonia VAE")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

##################################################################################################################
##################################################################################################################
##################################################################################################################
# Plot the distances
plt.figure()
plt.plot(epochs, ae_distances, "b-o", label="Training AE")
plt.plot(epochs, vae_distances, "b--s", label="Training VAE")
plt.plot(
    epochs,
    ae_val_healthy_distances,
    "g-o",
    label="Validation Healthy AE",
)
plt.plot(
    epochs,
    vae_val_healthy_distances,
    "g--s",
    label="Validation Healthy VAE",
)
plt.plot(
    epochs,
    ae_val_pneumonia_distances,
    "r-o",
    label="Validation Pneumonia AE",
)
plt.plot(
    epochs,
    vae_val_pneumonia_distances,
    "r--s",
    label="Validation Pneumonia VAE",
)
plt.title("Training and validation distances")
plt.xlabel("Epochs")
plt.ylabel("Distance")
plt.legend()

##################################################################################################################
##################################################################################################################
##################################################################################################################
# Run the models with the test set

aeGuesses = np.empty(0)
vaeGuesses = np.empty(0)

modelAE.eval()
modelVAE.eval()
with torch.no_grad():
    for _, data in enumerate(test_loader, 0):
        inputs_test, _ = data
        inputs_test = inputs_test.to(device)
        outputs_test = modelAE(inputs_test).view(-1, 1, 28, 28)

        # Add if they are healthy or pneumonia
        errors = (
            torch.sum(torch.abs(outputs_test - inputs_test), dim=(1, 2, 3))
            .cpu()
            .numpy()
        )
        predictions = (errors >= best_threshold_ae).astype(int)
        aeGuesses = np.append(aeGuesses, predictions)

        outputs_test, mean, logvar = modelVAE(inputs_test)
        outputs_test = outputs_test.view(-1, 1, 28, 28)

        # Add if they are healthy or pneumonia
        errors = (
            torch.sum(torch.abs(outputs_test - inputs_test), dim=(1, 2, 3))
            .cpu()
            .numpy()
        )
        predictions = (errors >= best_threshold_vae).astype(int)
        vaeGuesses = np.append(vaeGuesses, predictions)

# Show the confusion matrix
from sklearn.metrics import confusion_matrix

cm_ae = confusion_matrix(labels_test, aeGuesses)
cm_vae = confusion_matrix(labels_test, vaeGuesses)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cm_ae, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix AE")
plt.colorbar()
plt.xticks(np.arange(2), ["Healthy", "Pneumonia"])
plt.yticks(np.arange(2), ["Healthy", "Pneumonia"])
plt.ylabel("True label")
plt.xlabel("Predicted label")

# Anotate the values
for i in range(cm_ae.shape[0]):
    for j in range(cm_ae.shape[1]):
        plt.text(
            j, i, format(cm_ae[i, j], "d"), ha="center", va="center", color="black"
        )

plt.subplot(1, 2, 2)
plt.imshow(cm_vae, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix VAE")
plt.colorbar()
plt.xticks(np.arange(2), ["Healthy", "Pneumonia"])
plt.yticks(np.arange(2), ["Healthy", "Pneumonia"])
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

# Anotate the values
for i in range(cm_vae.shape[0]):
    for j in range(cm_vae.shape[1]):
        plt.text(
            j, i, format(cm_vae[i, j], "d"), ha="center", va="center", color="black"
        )

##################################################################################################################
##################################################################################################################
##################################################################################################################


plt.show()
