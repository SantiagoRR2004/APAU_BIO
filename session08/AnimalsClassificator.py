from torchvision import datasets
from CowsVsSheepsBasic import CowsVsSheepsBasic


class AnimalsClassificator(CowsVsSheepsBasic):
    num_classes = 3
    num_epochs = 50

    def eliminateClasses(
        self, train_dataset_orig: datasets.ImageFolder
    ) -> datasets.ImageFolder:
        ## Remove all instances that aren't mucca, pecora or cavallo
        # Get class-to-index mapping
        class_to_idx = train_dataset_orig.class_to_idx

        # Specify the class names you want to keep
        wanted_classes = ["mucca", "pecora", "cavallo"]

        # Create new class_to_idx mapping (e.g., {"mucca": 0, "pecora": 1, "cavallo": 2})
        new_class_to_idx = {
            class_name: idx for idx, class_name in enumerate(wanted_classes)
        }

        # Get the original indices for the wanted classes
        old_to_new_idx = {
            class_to_idx[class_name]: new_class_to_idx[class_name]
            for class_name in wanted_classes
        }

        # Filter and remap samples
        filtered_samples = [
            (path, old_to_new_idx[label])
            for (path, label) in train_dataset_orig.samples
            if label in old_to_new_idx
        ]
        filtered_targets = [label for (_, label) in filtered_samples]

        # Apply the filtered and remapped data
        train_dataset_orig.samples = filtered_samples
        train_dataset_orig.targets = filtered_targets
        train_dataset_orig.classes = wanted_classes
        train_dataset_orig.class_to_idx = new_class_to_idx

        return train_dataset_orig


if __name__ == "__main__":
    animalsClassificator = AnimalsClassificator()
    animalsClassificator.train()
    animalsClassificator.save_model(name="AnimalsClassificator.pth")
