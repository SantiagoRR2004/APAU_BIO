import torch
import matplotlib.pyplot as plt
import numpy as np


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


modelAE = AE()
modelAE.load_state_dict(
    torch.load("/home/leandro/models/AutoEncoders/_01_AE.pth", weights_only=True)
)
modelAE.eval()


# --------------------------------
# VAE definition
# --------------------------------


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder 784 -> 256 -> 48
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 48),
            torch.nn.ReLU(),
        )

        self.mean_layer = torch.nn.Linear(48, 2)
        self.logvar_layer = torch.nn.Linear(48, 2)

        # Decoder 2 -> 48 -> 256 -> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat  # , mean, logvar


modelVAE = VAE()
modelVAE.load_state_dict(
    torch.load("/home/leandro/models/AutoEncoders/_04_VAE.pth", weights_only=True)
)
modelVAE.eval()


# --------------------------------
# Plots AE and VAE latent spaces
# --------------------------------


def PlotLatentSpace(model, namefile, size, n_of_ticks, img_dim):
    xmin = -size
    xmax = size
    ymin = -size
    ymax = size
    figsize = 12
    figure = np.zeros((img_dim * n_of_ticks, img_dim * n_of_ticks))
    # linearly spaced coordinates corresponding to the 2D plot
    # of images classes in the latent space
    grid_x = np.linspace(xmin, xmax, n_of_ticks)
    grid_y = np.linspace(ymin, ymax, n_of_ticks)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.Tensor([[xi, yi]])
            x_decoded = model.decoder(z_sample).detach().numpy()
            images = x_decoded[0].reshape(img_dim, img_dim)
            figure[
                i * img_dim : (i + 1) * img_dim,
                j * img_dim : (j + 1) * img_dim,
            ] = images

    plt.figure(figsize=(figsize, figsize))
    start_range = img_dim // 2
    end_range = n_of_ticks * img_dim + start_range
    pixel_range = np.arange(start_range, end_range, img_dim)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    # plt.xticks(grid_x, grid_x)

    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="binary")
    plt.title("Latent space")
    plt.tight_layout()

    plt.savefig(namefile)


PlotLatentSpace(modelAE, "_05_LatentSpaceAE.png", size=10, n_of_ticks=20, img_dim=28)
PlotLatentSpace(modelVAE, "_05_LatentSpaceVAE.png", size=2, n_of_ticks=20, img_dim=28)
