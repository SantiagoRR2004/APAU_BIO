import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self, z_dim=100, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim * 4, kernel_size=3, stride=2),
            self.get_generator_block(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1
            ),
            self.get_generator_block(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            self.get_generator_final_block(
                hidden_dim, im_chan, kernel_size=4, stride=2
            ),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def get_generator_block(
        self, input_channel, output_channel, kernel_size, stride=1, padding=0
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channel, output_channel, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def get_generator_final_block(
        self, input_channel, output_channel, kernel_size, stride=1, padding=0
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channel, output_channel, kernel_size, stride, padding
            ),
            nn.Tanh(),
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


summary(Generator(100).to(device), (100,))
print(Generator(100))
