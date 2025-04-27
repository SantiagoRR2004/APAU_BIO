import torch
from CowsVsSheeps import CowsVsSheeps


class CowsVsSheepsBasic(CowsVsSheeps):
    def getCNN(self) -> torch.nn.Sequential:
        # Calculate the number of input features after flattening (channels * height * width)
        num_features = self.nChannels * self.img_dim * self.img_dim

        net = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.nChannels, 32, kernel_size=3
            ),  # self.nChannels × 32 × 3 × 3 + 32
            # (32, self.img_dim - 2, self.img_dim - 2)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d(kernel_size=2),
            # (32, self.img_dim - 4, self.img_dim - 4)
            torch.nn.Conv2d(32, 32, kernel_size=3),  # 32 × 32 × 3 × 3 + 32
            # (32, self.img_dim - 6, self.img_dim - 6)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d(kernel_size=2),
            # (32, self.img_dim - 8, self.img_dim - 8)
            torch.nn.Conv2d(32, 32, kernel_size=3),  # 32 × 32 × 3 × 3 + 32
            # (32, self.img_dim - 10, self.img_dim - 10)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d(kernel_size=2),
            # (32, self.img_dim - 12, self.img_dim - 12)
            torch.nn.Conv2d(32, 32, kernel_size=3),  # 32 × 32 × 3 × 3 + 32
            # (32, self.img_dim - 14, self.img_dim - 14)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(10368, self.num_classes),
        ).to(self.device)

        return net


if __name__ == "__main__":
    import os

    train = False
    cowVsSheep = CowsVsSheepsBasic()
    modelPath = os.path.join(cowVsSheep.currentDir, "CowVsSheep.pth")
    if train or not os.path.exists(modelPath):
        cowVsSheep.train()
        cowVsSheep.save_model(name=modelPath)
    else:
        cowVsSheep.load_model(path=modelPath)
        cowVsSheep.test()
