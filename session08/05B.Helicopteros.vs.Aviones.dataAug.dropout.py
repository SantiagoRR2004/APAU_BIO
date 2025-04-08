import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
import numpy as np
from PIL import Image
import random
import sys


def CreateMosaic(inputs, filename="helicoptersvsplanes.png"):
    selected_indices = random.sample(range(inputs.size(0)), 16)
    # print(selected_indices)
    to_pil = transforms.ToPILImage()
    images = [to_pil(inputs[i]) for i in selected_indices]
    img_width, img_height = images[0].size
    grid_image = Image.new("RGB", (img_width * 4, img_height * 4))

    # Paste images into the grid
    for i, img in enumerate(images):
        row = i // 4  # Get the row (0-3)
        col = i % 4  # Get the column (0-3)
        grid_image.paste(img, (col * img_width, row * img_height))

    # Save the resulting grid image
    grid_image.save(filename)


dataset_dir = "helicopter-vs-plane"
num_classes = 2
img_dim = 180
num_epochs = 50
batch_size = 64

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

custom_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(p=0.2),
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
    ]
)

train_dataset_orig = datasets.ImageFolder(dataset_dir, transform=custom_transform)

train_dataset, val_dataset = random_split(train_dataset_orig, [0.8, 0.2])

print("Length of training dataset:", len(train_dataset), "samples")
print("Length of validation dataset:", len(val_dataset), "samples")


train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, len(val_dataset), shuffle=False, num_workers=4)

net = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(32, 64, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(64, 128, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(128, 256, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(256, 256, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(12544, num_classes),
    # torch.nn.Softmax(dim=1)
).to(device)

print(net)

total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

loss_v = np.empty(0)
loss_val_v = np.empty(0)
accuracy_v = np.empty(0)
accuracy_val_v = np.empty(0)

for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if epoch == 0 and i == 0:
            CreateMosaic(inputs)
            # sys.exit(0)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        # # print statistics
        train_loss += batch_loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    net.eval()
    with torch.no_grad():
        data_iter = iter(val_loader)
        inputs_val, labels_val = next(data_iter)
        inputs_val = inputs_val.to(device)
        labels_val = labels_val.to(device)
        outputs_val = net(inputs_val)
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

torch.save(net.state_dict(), "helicopters-planes-droptout.pth")
print("modelo saved")

import matplotlib.pyplot as plt

num_epochs_stop = len(loss_val_v)
epochs = range(1, num_epochs_stop + 1)
plt.figure()
plt.plot(epochs, loss_v, "b-o", label="Training ")
plt.plot(epochs, loss_val_v, "r-o", label="Validation ")
plt.title("Training and validation loss (data aug. dropout)")
plt.xlabel("Epochs")
plt.ylim((0, 2))
plt.legend()
plt.savefig("05B.Helicopters.vs.planes.dataAug.dropout.Loss.png")

accuracy_v = accuracy_v[0:num_epochs_stop]
accuracy_val_v = accuracy_val_v[0:num_epochs_stop]
plt.figure()
plt.plot(epochs, accuracy_v, "b-o", label="Training ")
plt.plot(epochs, accuracy_val_v, "r-o", label="Validation ")
plt.title("Training and validation accuracy (data aug. dropout)")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("05B.Helicopters.vs.planes.dataAug.dropout.Accuracy.png")
