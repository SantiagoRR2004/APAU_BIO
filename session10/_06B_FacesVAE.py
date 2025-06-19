import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random, sys
import matplotlib.pyplot as plt
import numpy as np
import os


currentDirectory = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(currentDirectory, "img_align_celeba")
INPUT_DIM = (128, 128, 3)
BATCH_SIZE = 512


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

custom_transform = transforms.Compose(
    [transforms.Resize(INPUT_DIM[:2]), transforms.ToTensor()]
)

dataset = datasets.ImageFolder(DATASET_DIR, transform=custom_transform)
print("Length of dataset:", len(dataset), "images")
dataset_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)


# --------------------------------
# Shows a random image
# --------------------------------

num = random.randrange(len(dataset))

fig, ax = plt.subplots(nrows=1, ncols=1)
img_to_show = dataset[num][0].permute(1, 2, 0).numpy()
ax.imshow(img_to_show)
plt.tight_layout()
plt.savefig(os.path.join(currentDirectory, "_06_RandomFace.png"))


# --------------------------------
# VAE definition
# --------------------------------


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (3,128,128) -> (32,64,64) -> (64,32,32) -> (64,16,16) -> (64,8,8) -> 4096 -> 200
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
        )

        self.mean_layer = torch.nn.Linear(4096, 200)
        self.logvar_layer = torch.nn.Linear(4096, 200)

        # Decoder 200 -> 4096 -> (64,8,8) -> (64,16,16) -> (64,32,32) -> (32,64,64) -> (3,128,128)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(200, 4096),
            torch.nn.Unflatten(1, (64, 8, 8)),
            torch.nn.ConvTranspose2d(
                64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std).to(device)
        z = mean + std * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


model = VAE().to(device)
print("\nENCODER:")
print(model.encoder)
total_params = sum(p.numel() for p in model.encoder.parameters())
total_params += sum(p.numel() for p in model.mean_layer.parameters())
total_params += sum(p.numel() for p in model.logvar_layer.parameters())
print(f"Number of parameters in encoder: {total_params}")

print("\nDECODER:")
print(model.decoder)
total_params = sum(p.numel() for p in model.decoder.parameters())
print(f"Number of parameters in decoder: {total_params}")

print()


# --------------------------------
# VAE training
# --------------------------------


def loss_function(recon_x, x, mu, log_var):
    recon_loss = torch.sqrt(torch.nn.MSELoss()(recon_x, x))
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Total loss: reconstruction + KL regularization
    return recon_loss + 1.0e-6 * kl_loss


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

model.train()
for epoch in range(num_epochs):
    print("Epoch {:02d}/{:02d} [".format(epoch + 1, num_epochs), end="", flush=True)
    total_loss = 0.0
    total_loss_val = 0.0
    for i, data in enumerate(dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if i % 20 == 0:
            print("=", end="", flush=True)
        inputs, _ = data
        inputs = inputs.to(device)
        # print(inputs.shape)
        # sys.exit(0)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs, mu, log_var = model(inputs)
        loss = loss_function(outputs, inputs, mu, log_var)
        loss.backward()
        optimizer.step()

        # # statistics after a batch
        total_loss += loss.item()

    average_loss = total_loss / len(dataset_loader)
    print("] - loss {:.4f}".format(average_loss))

torch.save(model.state_dict(), os.path.join(currentDirectory, "_06_VAE_Faces.pth"))
