import torch
import torch.nn as nn
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Unflatten(1, (im_chan, 28, 28)),  # Add this line
            self.get_critic_block(im_chan, hidden_dim * 4, kernel_size=4, stride=2),
            self.get_critic_block(
                hidden_dim * 4,
                hidden_dim * 8,
                kernel_size=4,
                stride=2,
            ),
            self.get_critic_final_block(
                hidden_dim * 8,
                1,
                kernel_size=4,
                stride=2,
            ),
        )

    def get_critic_block(
        self, input_channel, output_channel, kernel_size, stride=1, padding=0
    ):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def get_critic_final_block(
        self, input_channel, output_channel, kernel_size, stride=1, padding=0
    ):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
        )

    def forward(self, image):
        x = self.disc(image)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 1)
        return torch.sigmoid(x)


summary(Discriminator().to(device), (784,))
print(Discriminator())
