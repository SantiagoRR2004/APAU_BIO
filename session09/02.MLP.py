import torch
import os
import kagglehub
import numpy as np
from matplotlib import pyplot as plt
import kerasReplacement
import sys

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    print("ERROR: no GPU available")
    sys.exit(0)

# --------------------------------
# Carga de datos
# --------------------------------

fname = kagglehub.dataset_download("stytch16/jena-climate-2009-2016")
fname = os.path.join(fname, "jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print("Campos:", header)
print("Número de registros:", len(lines))

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]  # Column 1 is the temperature array
    raw_data[i, :] = values[
        :
    ]  # All columns (including temperature) is the "raw_data" array
print("Valores temperatura:", temperature)
print("Dimensión valores temperatura:", temperature.shape)
print("Dimensión valores datos:", raw_data.shape)


# --------------------------------
# Plot de datos
# --------------------------------

# figure = plt.figure(figsize=(8,8))
# axes_temp = figure.add_subplot(221)
# axes_temp.plot(range(len(temperature)), temperature, color="red")
# axes_temp.set(ylabel='degC', xlabel='time', title="Temperatura")
# axes_temp = figure.add_subplot(222)
# axes_temp.plot(range(len(raw_data[:, 0:1])), raw_data[:, 0:1], color="blue")
# axes_temp.set(ylabel='mbar', xlabel='data', title="Presión")
# axes_temp = figure.add_subplot(223)
# axes_temp.plot(range(len(raw_data[:, 10:11])), raw_data[:, 10:11], color="green")
# axes_temp.set(ylabel='g/m**3', xlabel='data', title="Densidad aire")
# axes_temp = figure.add_subplot(224)
# axes_temp.plot(range(len(raw_data[:, 11:12])), raw_data[:, 11:12], color="brown")
# axes_temp.set(ylabel='m/s', xlabel='data', title="Velocidad viento")
# figure.tight_layout()
# figure.savefig("JenaClimate.png")

# --------------------------------
# División dataset y normalización
# --------------------------------

num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)


mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
# Nota... se normaliza toda la entrada. La temperatura a la salida no


# --------------------------------
# Construcción de los datasets
# --------------------------------

# Prueba de dataset sencillo

# int_sequence = np.arange(10)  # Our raw data [0 1 2 3 4 5 6 7 8 9]
# dummy_dataset = kerasReplacement.timeseries_dataset_from_array(
#     data=int_sequence[:-3],   # it will be [0 1 2 3 4 5 6] excluding the last three
#     targets=int_sequence[3:], # Target for the sequence that starts at data[N] will be data[N+3]
#     sequence_length=3,        # The sequences will be 3 steps long: [0 1 2], [1 2 3], [2 3 4], ...
#     batch_size=2,             # The sequences will be batched in batches of size 2
# )

# for inputs, targets in dummy_dataset:
#     for i in range(inputs.shape[0]):
#         print([int(x) for x in inputs[i]], int(targets[i]))

# Construcción de train, test y val datasets

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = kerasReplacement.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
)

val_dataset = kerasReplacement.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=num_val_samples,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)

# test_dataset = kerasReplacement.timeseries_dataset_from_array(
#     raw_data[:-delay],
#     targets=temperature[delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=False,
#     batch_size=num_test_samples,
#     start_index=num_train_samples + num_val_samples)

# Comprobamos la forma de los batches
# for samples, targets in train_dataset:
#     print("samples shape:", samples.shape)
#     print("targets shape:", targets.shape)
#     break


# --------------------------------
# MLP
# --------------------------------


net = torch.nn.Sequential(
    torch.nn.Linear(sequence_length * 14, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
).to(device)

print(net)

total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

num_epochs = 10
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

loss_v = np.empty(0)
loss_val_v = np.empty(0)
mae_v = np.empty(0)
mae_val_v = np.empty(0)

for epoch in range(num_epochs):
    train_loss = 0.0
    train_mae = 0.0
    samples_seen_train = 0
    val_mae = 0.0
    samples_seen_val = 0
    net.train()
    for samples, targets in train_dataset:
        # print(samples.shape)
        samples_flat = samples.numpy().reshape(samples.shape[0], -1)
        targets_flat = targets.numpy().reshape(targets.shape[0], 1)
        # print(samples_flat.shape)
        torch_samples = torch.from_numpy(samples_flat).float().to(device)
        torch_targets = torch.from_numpy(targets_flat).float().to(device)
        optimizer.zero_grad()
        outputs = net(torch_samples)
        loss = criterion(outputs, torch_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += np.sum(
            np.abs(targets.numpy().flatten() - outputs.detach().cpu().numpy().flatten())
        )
        samples_seen_train += len(targets.numpy())
        # print(samples_seen)

    net.eval()
    with torch.no_grad():
        for samples, targets in val_dataset:  # batch_size = size(val_dataset)
            samples_flat = samples.numpy().reshape(samples.shape[0], -1)
            targets_flat = targets.numpy().reshape(targets.shape[0], 1)
            torch_samples = torch.from_numpy(samples_flat).float().to(device)
            torch_targets = torch.from_numpy(targets_flat).float().to(device)
            outputs = net(torch_samples)
            loss = criterion(outputs, torch_targets)
            val_loss = loss.item()
            val_mae += np.sum(
                np.abs(
                    targets.numpy().flatten() - outputs.detach().cpu().numpy().flatten()
                )
            )
            samples_seen_val += len(targets.numpy())

    train_loss = train_loss / len(train_dataset)
    train_mae = train_mae / samples_seen_train
    val_loss = val_loss / len(val_dataset)
    val_mae = val_mae / samples_seen_val

    print(
        "Epoch {:02d}: loss {:.4f} - train mae {:.4f} - val. loss {:.4f} - val. mae {:.4f}".format(
            epoch + 1, train_loss, train_mae, val_loss, val_mae
        )
    )

    loss_v = np.append(loss_v, train_loss)
    loss_val_v = np.append(loss_val_v, val_loss)
    mae_v = np.append(mae_v, train_mae)
    mae_val_v = np.append(mae_val_v, val_mae)

import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs, loss_v, "b-o", label="Training ")
plt.plot(epochs, loss_val_v, "r-o", label="Validation ")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("02.MLP.Loss.png")

plt.figure()
plt.plot(epochs, mae_v, "b-o", label="Training ")
plt.plot(epochs, mae_val_v, "r-o", label="Validation ")
plt.title("Training and validation MAE")
plt.xlabel("Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("02.MLP.MAE.png")
