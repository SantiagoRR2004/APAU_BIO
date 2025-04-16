import torch
from CowsVsSheeps import CowsVsSheeps


class CowsVsSheepsBasic(CowsVsSheeps):
    def getCNN(self) -> torch.nn.Sequential:
        # Calculate the number of input features after flattening (channels * height * width)
        num_features = 3 * self.img_dim * self.img_dim

        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3),  # 32 × 3 × 3 × 3 + 32
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


if __name__ == "__main__":
    CowsVsSheepsBasic().train()
