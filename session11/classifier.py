import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    num_classes = 10
    img_dim = 28
    num_epochs = 50
    batch_size = 128
    nChannels = 1

    def __init__(self):
        super(Classifier, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.net = self.getCNN()

        self.class_names = []

    def forward(self, image):
        return self.net(image)

    def getCNN(self) -> torch.nn.Sequential:
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1152, 10),
        ).to(self.device)

        return net

    def train(self, train_loader):

        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        loss_v = np.empty(0)
        accuracy_v = np.empty(0)

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

            print(
                "Epoch {:02d}: loss {:.4f} - accuracy {:.4f} ".format(
                    epoch + 1, train_loss, accuracy
                )
            )

            loss_v = np.append(loss_v, train_loss)
            accuracy_v = np.append(accuracy_v, accuracy)

    def save_model(self, name: str = "Classifier.pth"):
        torch.save(self.net.state_dict(), name)

    def plot_image(self, image: torch.Tensor):
        # Convert the tensor to a numpy array
        image = image.cpu().numpy()
        # Transpose the dimensions to (height, width, channels)
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image, cmap="gray_r" if self.nChannels == 1 else None)

    def load_model(self, path: str = "Classifier.pth"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def predict(self, test_loader):

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
                total_samples_val += labels.size(0)

        inputs_val = torch.cat(inputs_val, dim=0)
        labels_val = torch.cat(labels_val, dim=0)
        outputs_val = torch.cat(outputs_val, dim=0)
        predicted_val = torch.cat(predicted_val, dim=0)

        # Set up the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

        # Loop through all images and show one every 10 seconds
        for i in range(len(inputs_val)):
            ax.clear()  # Clear previous plot
            self.plot_image(inputs_val[i])  # Pass ax to your plot_image function
            predicted_class = self.class_names[predicted_val[i].item()]
            ax.set_title(f"Predicted: {predicted_class}")

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


summary(Classifier().to(device), (1, 28, 28))
print(Classifier())
