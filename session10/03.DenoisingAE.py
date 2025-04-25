import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import sys
import matplotlib.pyplot as plt
import numpy as np

myNoiseFactor = 0.4


# ------------------------------------------
# Code to add noise to a batch of images
# ------------------------------------------

def add_noise_to_batch(images, noise_factor=0.5):
    noise = torch.randn_like(images) * noise_factor  # Generate noise for the entire batch
    noisy_images = images + noise  # Add noise to the batch of images
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # Clamp values to be between 0 and 1
    return noisy_images
    
# ------------------------------------------


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    print("ERROR: no GPU available")
    sys.exit(0)

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

mnist_dataset = datasets.MNIST(root='/tmp/data', download=True, transform = custom_transform)
print("Length of dataset:", len(mnist_dataset))

train_data = Subset(mnist_dataset, list(range(0, 50000)))
val_data = Subset(mnist_dataset, list(range(50000,len(mnist_dataset))))
## Print the length of train and validation datasets
print("Length of train Dataset: ", len(train_data))
print("Length of validation Dataset: ", len(val_data))

batch_size = 128

train_loader = DataLoader(train_data, batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size , shuffle = False)


# ---------------------------------------------
# Show some original and degraded digits
# ---------------------------------------------

n_to_show = 10
input_idx = np.random.choice(range(len(train_data)), n_to_show)
input_images = [ train_data[i][0] for i in input_idx ]
input_labels = [ train_data[i][1] for i in input_idx ]
val_imgs_to_show = torch.stack(input_images)

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    img = val_imgs_to_show[i].reshape((28,28))
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.set_title(input_labels[i])
    fig.suptitle('Train original digits')
    ax.imshow(img, cmap='binary')
fig.tight_layout()
fig.savefig("03.TrainCleanDigits.png")

val_imgs_to_show_noisy = add_noise_to_batch(val_imgs_to_show, myNoiseFactor)
fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    img = val_imgs_to_show_noisy[i].reshape((28,28))
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.set_title(input_labels[i])
    fig.suptitle('Train noisy digits')
    ax.imshow(img, cmap='binary')
fig.tight_layout()
fig.savefig("03.TrainNoisyDigits.png")


# --------------------------------
# AE definition
# --------------------------------

model =  torch.nn.Sequential(
            torch.nn.Linear(784, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 784),
            torch.nn.Sigmoid()
        ).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")



# --------------------------------
# AE training
# --------------------------------

criterion = torch.nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001)
num_epochs = 30

loss_v = np.empty(0)
loss_val_v = np.empty(0)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_loss_val = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        clean_digits, _ = data
        noisy_digits = add_noise_to_batch(clean_digits, myNoiseFactor)
        clean_digits = clean_digits.to(device)
        noisy_digits = noisy_digits.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(noisy_digits)
        loss = criterion(outputs, clean_digits)
        loss.backward()
        optimizer.step()

        # # statistics after a batch
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            clean_digits_val, _ = data
            noisy_digits_val = add_noise_to_batch(clean_digits_val, myNoiseFactor)
            clean_digits_val = clean_digits_val.to(device)
            noisy_digits_val = noisy_digits_val.to(device)
            outputs_val = model(noisy_digits_val)
            loss_val = criterion(outputs_val, clean_digits_val)
            total_loss_val += loss_val.item()

    average_loss = total_loss / len(train_loader)
    average_loss_val = total_loss_val / len(val_loader)
    loss_v = np.append(loss_v, average_loss)
    loss_val_v = np.append(loss_val_v, average_loss_val)


    print("Epoch {:02d}: loss {:.4f} - val. loss {:.4f}".format(epoch+1, average_loss, average_loss_val))

torch.save(model.state_dict(), "/home/leandro/models/AutoEncoders/03.DenoisingAE.pth")


# --------------------------------
# Plot loss
# --------------------------------

epochs = range(1, num_epochs+1)
plt.figure()
plt.plot(epochs, loss_v, 'b-o', label='Training ')
plt.plot(epochs, loss_val_v, 'r-o', label='Validation ') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()

plt.savefig("03.DenoisingAE.Loss.png")




# ----------------------------------------------------
# Show some validation clean, noisy and cleaned digits
# ----------------------------------------------------

n_to_show = 10
input_idx = np.random.choice(range(len(val_data)), n_to_show)
input_images = [ val_data[i][0] for i in input_idx ]
input_labels = [ val_data[i][1] for i in input_idx ]
val_imgs_to_show = torch.stack(input_images)

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    img = val_imgs_to_show[i].reshape((28,28))
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.set_title(input_labels[i])
    fig.suptitle('Validation original digits')
    ax.imshow(img, cmap='binary')
fig.tight_layout()
fig.savefig("03.ValCleanDigits.png")

val_imgs_to_show_noisy = add_noise_to_batch(val_imgs_to_show, myNoiseFactor)
fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    img = val_imgs_to_show_noisy[i].reshape((28,28))
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.set_title(input_labels[i])
    fig.suptitle('Validation noisy digits')
    ax.imshow(img, cmap='binary')
fig.tight_layout()
fig.savefig("03.ValNoisyDigits.png")

val_imgs_to_show_cleaned = model(val_imgs_to_show_noisy.to(device))

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    img = val_imgs_to_show_cleaned[i].cpu().detach().numpy().reshape((28,28))
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.set_title(input_labels[i])
    fig.suptitle('Validation cleaned digits')
    ax.imshow(img, cmap='binary')
fig.tight_layout()
fig.savefig("03.ValCleanedDigits.png")

sys.exit(0)





