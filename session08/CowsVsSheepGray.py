import torch
from CowsVsSheepsBasic import CowsVsSheepsBasic
from torchvision import transforms

# En lugar de tres canales
# Promediamos a uno (blanco y negro)


class CowsVsSheepsGray(CowsVsSheepsBasic):
    nChannels = 1

    def get_transform(self):
        def rgb_to_grayscale(x):
            return torch.mean(x, dim=0, keepdim=True)

        return transforms.Compose(
            [
                transforms.Resize((self.img_dim, self.img_dim)),
                transforms.ToTensor(),
                transforms.Lambda(rgb_to_grayscale),
            ]
        )


if __name__ == "__main__":
    cowVsSheepGray = CowsVsSheepsGray()
    cowVsSheepGray.train()
    cowVsSheepGray.save_model(name="CowVsSheepGrey.pth")
