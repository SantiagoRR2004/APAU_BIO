import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import sys, random
import matplotlib.pyplot as plt
import numpy as np
import os


# --------------------------------
# AE definition
# --------------------------------


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder 784 -> 256 -> 48 -> 2
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 2),
        )

        # Decoder 2 -> 48 -> 256 -> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU available:", torch.cuda.get_device_name(0))
    else:
        print("ERROR: no GPU available")
        sys.exit(0)

    currentDirectory = os.path.dirname(os.path.abspath(__file__))

    custom_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )

    mnist_dataset = datasets.MNIST(
        root="/tmp/data", download=True, transform=custom_transform
    )
    print("Length of dataset:", len(mnist_dataset))
    print("Length of first vector in dataset: ", mnist_dataset[0][0].shape)
    print("Label of first vector in dataset: ", mnist_dataset[0][1])

    train_data = Subset(mnist_dataset, list(range(0, 50000)))
    val_data = Subset(mnist_dataset, list(range(50000, len(mnist_dataset))))
    ## Print the length of train and validation datasets
    print("Length of train Dataset: ", len(train_data))
    print("Length of validation Dataset: ", len(val_data))

    batch_size = 128

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, len(val_data), shuffle=False, num_workers=2)

    # --------------------------------
    # Shows a random image
    # --------------------------------

    # num = random.randrange(len(train_data))
    # fig, ax = plt.subplots( nrows=1, ncols=1 )
    # ax.imshow(train_data[num][0].reshape((28,28)), cmap='binary')
    # ax.set_title("Digit: {}".format(train_data[num][1]))
    # plt.tight_layout()
    # plt.savefig(os.path.join(currentDirectory, "_01_RandomDigit.png"))

    model = AE().to(device)

    print("\nENCODER:")
    print(model.encoder)
    total_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters: {total_params}")

    print("\nDECODER:")
    print(model.decoder)
    total_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Number of parameters: {total_params}")

    # --------------------------------
    # AE training
    # --------------------------------

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    num_epochs = 30

    loss_v = np.empty(0)
    loss_val_v = np.empty(0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_loss_val = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            inputs = inputs.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # # statistics after a batch
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs_val, _ = data
                inputs_val = inputs_val.to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, inputs_val)
                total_loss_val += loss_val.item()

        average_loss = total_loss / len(train_loader)
        average_loss_val = total_loss_val / len(val_loader)
        loss_v = np.append(loss_v, average_loss)
        loss_val_v = np.append(loss_val_v, average_loss_val)

        print(
            "Epoch {:02d}: loss {:.4f} - val. loss {:.4f}".format(
                epoch + 1, average_loss, average_loss_val
            )
        )

    torch.save(model.state_dict(), os.path.join(currentDirectory, "_01_AE.pth"))

    # --------------------------------
    # Plot loss
    # --------------------------------

    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, loss_v, "b-o", label="Training ")
    plt.plot(epochs, loss_val_v, "r-o", label="Validation ")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(currentDirectory, "_01_UndercompleteAE.Loss.png"))

    # ---------------------------------------------
    # Show some original and reconstructed digits
    # ---------------------------------------------

    n_to_show = 10
    input_idx = np.random.choice(range(len(val_data)), n_to_show)
    input_images = [val_data[i][0] for i in input_idx]
    input_labels = [val_data[i][1] for i in input_idx]

    input_batch = torch.stack(input_images).to(device)
    output_batch = model(input_batch)

    # print(input_labels)
    # print(len(output_batch))

    fig = plt.figure(figsize=(15, 2))
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    for i in range(n_to_show):
        img = input_batch[i].cpu().reshape((28, 28))
        ax = fig.add_subplot(1, n_to_show, i + 1)
        ax.axis("off")
        ax.set_title(input_labels[i])
        fig.suptitle("Original digits")
        ax.imshow(img, cmap="binary")
    fig.tight_layout()
    fig.savefig(os.path.join(currentDirectory, "_01_OriginalDigits.png"))

    fig = plt.figure(figsize=(15, 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n_to_show):
        img = output_batch[i].detach().cpu().reshape((28, 28))
        ax = fig.add_subplot(1, n_to_show, i + 1)
        ax.axis("off")
        ax.set_title(input_labels[i])
        fig.suptitle("Reconstructed digits")
        ax.imshow(img, cmap="binary")
    fig.tight_layout()
    fig.savefig(os.path.join(currentDirectory, "_01_ReconstructedDigits.png"))

    # ---------------------------------------------
    # Show training points in latent space
    # ---------------------------------------------

    n_to_show = 5000
    grid_size = 20
    figsize = 8
    example_idx = np.random.choice(range(len(train_data)), n_to_show)
    example_images = [train_data[i][0] for i in example_idx]
    example_labels = [train_data[i][1] for i in example_idx]

    example_batch = torch.stack(example_images).to(device)
    example_points = model.encoder(example_batch).detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(
        example_points[:, 0],
        example_points[:, 1],
        cmap="tab10",
        c=example_labels,
        alpha=0.5,
        s=2,
    )
    plt.colorbar(values=range(10), ticks=range(10))
    plt.tight_layout()
    plt.savefig(os.path.join(currentDirectory, "_01_LatentSpace.png"))
