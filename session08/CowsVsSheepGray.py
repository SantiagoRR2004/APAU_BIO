import torch
from CowsVsSheeps import CowsVsSheeps
from torchvision import transforms

# En lugar de tres canales
# Promediamos a uno (blanco y negro)


class CowsVsSheepsGray(CowsVsSheeps):
    def getCNN(self) -> torch.nn.Sequential:
        # Calculate the number of input features after flattening (channels * height * width)
        num_features = 1 * self.img_dim * self.img_dim

        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),  # 32 × 3 × 3 × 3 + 32
            # (32, self.img_dim - 2, self.img_dim - 2)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # (32, self.img_dim - 4, self.img_dim - 4)
            torch.nn.Conv2d(32, 32, kernel_size=3),  # 32 × 32 × 3 × 3 + 32
            # (32, self.img_dim - 6, self.img_dim - 6)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # (32, self.img_dim - 8, self.img_dim - 8)
            torch.nn.Conv2d(32, 32, kernel_size=3),  # 32 × 32 × 3 × 3 + 32
            # (32, self.img_dim - 10, self.img_dim - 10)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # (32, self.img_dim - 12, self.img_dim - 12)
            torch.nn.Conv2d(32, 32, kernel_size=3),  # 32 × 32 × 3 × 3 + 32
            # (32, self.img_dim - 14, self.img_dim - 14)
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(10368, self.num_classes),
        ).to(self.device)

        return net

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
