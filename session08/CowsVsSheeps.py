import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import os
import numpy as np


class CowsVsSheeps:
    num_classes = 2
    img_dim = 180
    num_epochs = 50
    batch_size = 64
    currentDir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = os.path.join(currentDir, "animals10", "raw-img")

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        custom_transform = transforms.Compose(
            [
                transforms.Resize((self.img_dim, self.img_dim)),
                transforms.ToTensor(),
            ]
        )

        train_dataset_orig = datasets.ImageFolder(
            self.dataset_dir, transform=custom_transform
        )

        train_dataset_orig = self.eliminateClasses(train_dataset_orig)

        self.train_dataset, self.val_dataset = random_split(
            train_dataset_orig, [0.8, 0.2]
        )

        # Print the number of classes of self.train_dataset
        print("Length of training dataset:", len(self.train_dataset), "samples")
        print("Length of validation dataset:", len(self.val_dataset), "samples")

        self.net = self.getCNN()

    def eliminateClasses(
        self, train_dataset_orig: datasets.ImageFolder
    ) -> datasets.ImageFolder:
        ## Remove all instances that aren't mucca or pecora
        # Get class-to-index mapping
        class_to_idx = train_dataset_orig.class_to_idx

        # Specify the class names you want to keep
        wanted_classes = ["mucca", "pecora"]

        # Create new class_to_idx mapping (e.g., {"mucca": 0, "pecora": 1})
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

    def getCNN(self) -> torch.nn.Sequential:
        # Calculate the number of input features after flattening (channels * height * width)
        num_features = 3 * self.img_dim * self.img_dim

        # Create the sequential model
        net = torch.nn.Sequential(
            torch.nn.Flatten(),  # Flatten the input tensor
            torch.nn.Linear(num_features, self.num_classes),  # Output to num_classes
        ).to(self.device)

        return net

    def train(self):
        train_loader = DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            self.val_dataset, len(self.val_dataset), shuffle=False, num_workers=4
        )

        # loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        loss_v = np.empty(0)
        loss_val_v = np.empty(0)
        accuracy_v = np.empty(0)
        accuracy_val_v = np.empty(0)

        for epoch in range(self.num_epochs):
            self.net.train()
            train_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = self.net(inputs)
                batch_loss = criterion(outputs, labels)
                batch_loss.backward()
                optimizer.step()

                # # print statistics
                train_loss += batch_loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            self.net.eval()
            with torch.no_grad():
                data_iter = iter(val_loader)
                inputs_val, labels_val = next(data_iter)
                inputs_val = inputs_val.to(self.device)
                labels_val = labels_val.to(self.device)
                outputs_val = self.net(inputs_val)
                _, predicted = torch.max(outputs_val, 1)
                correct_predictions_val = (predicted == labels_val).sum().item()
                total_samples_val = labels_val.size(0)
                val_loss = criterion(outputs_val, labels_val).item() / len(val_loader)

            accuracy = correct_predictions / total_samples
            val_accuracy = correct_predictions_val / total_samples_val
            train_loss = train_loss / len(train_loader)

            print(
                "Epoch {:02d}: loss {:.4f} - accuracy {:.4f} - val. loss {:.4f} - val. acc. {:.4f}".format(
                    epoch + 1, train_loss, accuracy, val_loss, val_accuracy
                )
            )

            loss_v = np.append(loss_v, train_loss)
            loss_val_v = np.append(loss_val_v, val_loss)
            accuracy_v = np.append(accuracy_v, accuracy)
            accuracy_val_v = np.append(accuracy_val_v, val_accuracy)


if __name__ == "__main__":
    CowsVsSheeps().train()
