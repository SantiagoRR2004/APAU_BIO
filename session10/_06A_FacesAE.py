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
# AE definition
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


model = AE().to(device)
print("\nENCODER:")
print(model.encoder)
total_params = sum(p.numel() for p in model.encoder.parameters())
print(f"Number of parameters in encoder: {total_params}")

print("\nDECODER:")
print(model.decoder)
total_params = sum(p.numel() for p in model.decoder.parameters())
print(f"Number of parameters in decoder: {total_params}")

print()

# test_output = model.encoder(dataset[0][0])
# print(test_output.shape)
# sys.exit(0)


# --------------------------------
# AE training
# --------------------------------


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


criterion = RMSELoss()
# criterion = torch.nn.MSELoss()
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
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # # statistics after a batch
        total_loss += loss.item()

    average_loss = total_loss / len(dataset_loader)
    print("] - loss {:.4f}".format(average_loss))

torch.save(model.state_dict(), os.path.join(currentDirectory, "_06_AE_Faces.pth"))
