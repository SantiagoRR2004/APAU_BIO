import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import sys

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
print("Length of first vector in dataset: ", mnist_dataset[0][0].shape)
print("Label of first vector in dataset: ", mnist_dataset[0][1])

train_data, val_data = random_split(mnist_dataset, [50000, 10000])
## Print the length of train and validation datasets
print("Length of Train Dataset: ", len(train_data))
print("Length of Test Dataset: ", len(val_data))

batch_size = 128

train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=2)
val_loader = DataLoader(val_data, len(val_data) , shuffle = False, num_workers=2)

net = torch.nn.Sequential(
      torch.nn.Linear(784, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, 10),
      torch.nn.Softmax(dim=1)
      ).to(device)

print(net)

num_epochs = 50

optimizer = torch.optim.RMSprop(net.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # # statistics after a batch
        total_loss += loss.item()
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

    accuracy = correct_predictions / total_samples
    accuracy_val = correct_predictions_val / total_samples_val
    average_loss = total_loss / len(train_loader)
    

    print("Epoch {:02d}: loss {:.4f} - accuracy {:.4f} - validation accuracy {:.4f}".format(epoch+1, average_loss, accuracy, accuracy_val))