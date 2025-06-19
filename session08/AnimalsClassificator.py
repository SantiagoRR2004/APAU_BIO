from torchvision import datasets
from CowsVsSheepsBasic import CowsVsSheepsBasic


class AnimalsClassificator(CowsVsSheepsBasic):
    num_classes = 10
    num_epochs = 50

    def eliminateClasses(
        self, train_dataset_orig: datasets.ImageFolder
    ) -> datasets.ImageFolder:
        return train_dataset_orig


if __name__ == "__main__":
    import os

    train = False
    animalsClassificator = AnimalsClassificator()
    modelPath = os.path.join(
        animalsClassificator.currentDir, "models", "AnimalsClassificator.pth"
    )
    if train or not os.path.exists(modelPath):
        animalsClassificator.train()
        animalsClassificator.save_model(name=modelPath)
    else:
        animalsClassificator.load_model(path=modelPath)
        animalsClassificator.test()
