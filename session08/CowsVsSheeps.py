import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
import os
import numpy as np


class CowsVsSheeps:
    num_classes = 2
    img_dim = 180
    num_epochs = 50
    batch_size = 64
    nChannels = 3
    currentDir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = os.path.join(currentDir, "animals10", "raw-img")

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        custom_transform = self.get_transform()

        train_dataset_orig = datasets.ImageFolder(
            self.dataset_dir, transform=custom_transform
        )

        train_dataset_orig = self.eliminateClasses(train_dataset_orig)

        self.class_names = train_dataset_orig.classes

        self.train_dataset, self.val_dataset = random_split(
            train_dataset_orig, [0.8, 0.2]
        )

        # Print the number of classes of self.train_dataset
        print("Length of training dataset:", len(self.train_dataset), "samples")
        print("Length of validation dataset:", len(self.val_dataset), "samples")

        self.net = self.getCNN()

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.img_dim, self.img_dim)),
                transforms.ToTensor(),
            ]
        )

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
        num_features = self.nChannels * self.img_dim * self.img_dim

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
            self.val_dataset, self.batch_size, shuffle=False, num_workers=4
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

            accuracy = correct_predictions / total_samples

            self.net.eval()
            val_loss_total = 0
            correct_predictions_val = 0
            total_samples_val = 0

            with torch.no_grad():
                for data in val_loader:
                    inputs_val, labels_val = data
                    inputs_val = inputs_val.to(self.device)
                    labels_val = labels_val.to(self.device)

                    outputs_val = self.net(inputs_val)
                    batch_loss = criterion(outputs_val, labels_val)
                    val_loss_total += batch_loss.item()

                    _, predicted = torch.max(outputs_val, 1)
                    correct_predictions_val += (predicted == labels_val).sum().item()
                    total_samples_val += labels_val.size(0)

            val_loss = val_loss_total / len(val_loader)
            val_accuracy = correct_predictions_val / total_samples_val

            print(
                "Epoch {:02d}: loss {:.4f} - accuracy {:.4f} - val. loss {:.4f} - val. acc. {:.4f}".format(
                    epoch + 1, train_loss, accuracy, val_loss, val_accuracy
                )
            )

            loss_v = np.append(loss_v, train_loss)
            loss_val_v = np.append(loss_val_v, val_loss)
            accuracy_v = np.append(accuracy_v, accuracy)
            accuracy_val_v = np.append(accuracy_val_v, val_accuracy)

    def save_model(self, name: str = "CowVsSheep.pth"):
        torch.save(self.net.state_dict(), name)

    def plot_image(self, image: torch.Tensor):
        # Convert the tensor to a numpy array
        image = image.cpu().numpy()
        # Transpose the dimensions to (height, width, channels)
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image, cmap="gray" if self.nChannels == 1 else None)

    def load_model(self, path: str = "CowVsSheep.pth"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def test(self):
        torch.manual_seed(0)
        test_loader = DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, num_workers=1
        )

        correct_predictions_val = 0
        total_samples_val = 0
        inputs_val = []
        labels_val = []
        outputs_val = []
        predicted_val = []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs_val.append(inputs)
                inputs = inputs.to(self.device)
                labels_val.append(labels)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                outputs_val.append(outputs)
                _, predicted = torch.max(outputs, 1)
                predicted_val.append(predicted)
                correct_predictions_val += (predicted == labels).sum().item()
                total_samples_val += labels.size(0)

        inputs_val = torch.cat(inputs_val, dim=0)
        labels_val = torch.cat(labels_val, dim=0)
        outputs_val = torch.cat(outputs_val, dim=0)
        predicted_val = torch.cat(predicted_val, dim=0)

        accuracy_val = correct_predictions_val / total_samples_val
        print("Test accuracy: {:.4f}".format(accuracy_val))

        # Set up the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

        # Loop through all images and show one every 10 seconds
        for i in range(len(inputs_val)):
            ax.clear()  # Clear previous plot
            self.plot_image(inputs_val[i])  # Pass ax to your plot_image function
            predicted_class = self.class_names[predicted_val[i].item()]
            actual_class = self.class_names[labels_val[i].item()]
            ax.set_title(f"Predicted: {predicted_class}, Actual: {actual_class}")

            probs = F.softmax(outputs_val[i], dim=0)

            # Update the class percentages dynamically
            class_percentages = {
                self.class_names[j]: probs[j].item() * 100
                for j in range(len(self.class_names))
            }

            # Update legend lines with new percentages
            legend_lines = [
                f"{class_name}: {percentage:.1f}%"
                for class_name, percentage in class_percentages.items()
            ]

            legend_patches = [
                mpatches.Patch(color="none", label=line) for line in legend_lines
            ]

            ax.legend(
                handles=legend_patches,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
                handlelength=0,
                handletextpad=0,
            )

            plt.tight_layout()
            plt.draw()  # Update the figure
            plt.pause(5)  # Pause for 5 seconds to show the image

        plt.ioff()  # Turn off interactive mode
        plt.show()


if __name__ == "__main__":
    train = False
    cowVsSheep = CowsVsSheeps()
    modelPath = os.path.join(cowVsSheep.currentDir, "CowVsSheepLineal.pth")
    if train or not os.path.exists(modelPath):
        cowVsSheep.train()
        cowVsSheep.save_model(name=modelPath)
    else:
        cowVsSheep.load_model(path=modelPath)
        cowVsSheep.test()
