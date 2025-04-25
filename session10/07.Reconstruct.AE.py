import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random, sys
import matplotlib.pyplot as plt
import numpy as np


DATASET_DIR = "/home/leandro/datasets/img_align_celeba"
INPUT_DIM = (128, 128, 3)
# BATCH_SIZE = 512


# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print("GPU available:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device('cpu')

custom_transform = transforms.Compose(
    [transforms.Resize(INPUT_DIM[:2]), transforms.ToTensor()]
)

dataset = datasets.ImageFolder(DATASET_DIR, transform=custom_transform)
print("Length of dataset:", len(dataset), "images")
# dataset_loader = DataLoader(dataset, BATCH_SIZE, shuffle = True)


# # --------------------------------
# # Shows a random image
# # --------------------------------

# num = random.randrange(len(dataset))

# fig, ax = plt.subplots( nrows=1, ncols=1 )
# img_to_show = dataset[num][0].permute(1,2,0).numpy()
# ax.imshow(img_to_show)
# plt.tight_layout()
# plt.savefig("06.RandomFace.png")


# --------------------------------
# AE loading
# --------------------------------


class AE(torch.nn.Module):
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
            torch.nn.Linear(4096, 200),
        )

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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model_AE = AE()
model_AE.load_state_dict(
    torch.load("/home/leandro/models/AutoEncoders/06.AE_Faces.pth", weights_only=True)
)
model_AE.eval()


print("\nAE model loaded from file")
total_params = sum(p.numel() for p in model_AE.parameters())
print(f"Number of parameters in AE: {total_params}")


# --------------------------------
# VAE loading
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
            # torch.nn.Linear(4096, 200),
        )

        self.mean_layer = torch.nn.Linear(4096, 400)
        self.logvar_layer = torch.nn.Linear(4096, 400)

        # Decoder 200 -> 4096 -> (64,8,8) -> (64,16,16) -> (64,32,32) -> (32,64,64) -> (3,128,128)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(400, 4096),
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


model_VAE = VAE()
model_VAE.load_state_dict(
    torch.load("/home/leandro/models/AutoEncoders/06.VAE_Faces.pth", weights_only=True)
)
model_VAE.eval()


print("\nVAE model loaded from file")
total_params = sum(p.numel() for p in model_VAE.parameters())
print(f"Number of parameters in VAE: {total_params}")


# --------------------------------
# Shows a random image
# --------------------------------

num = random.randrange(len(dataset))

fig, ax = plt.subplots(nrows=1, ncols=1)
img_to_show = dataset[num][0].permute(1, 2, 0).numpy()
ax.imshow(img_to_show)
plt.tight_layout()
plt.savefig("07.RandomFace_orig.png")

fig, ax = plt.subplots(nrows=1, ncols=1)
img_out = torch.stack([dataset[num][0]])
img_to_show = model_AE(img_out)[0].detach().permute(1, 2, 0).numpy()
ax.imshow(img_to_show)
plt.tight_layout()
plt.savefig("07.RandomFace_reconstr_AE.png")

fig, ax = plt.subplots(nrows=1, ncols=1)
img_out = torch.stack([dataset[num][0]])
img_to_show = model_VAE(img_out)[0][0].detach().permute(1, 2, 0).numpy()
ax.imshow(img_to_show)
plt.tight_layout()
plt.savefig("07.RandomFace_reconstr_VAE.png")


# test_output = model.encoder(dataset[0][0])
# print(test_output.shape)
# sys.exit(0)
