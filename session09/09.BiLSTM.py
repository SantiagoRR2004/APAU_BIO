import torch
import os
import kagglehub
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
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
# dummy_dataset = keras.utils.timeseries_dataset_from_array(
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

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)


# Comprobamos la forma de los batches
# for samples, targets in train_dataset:
#     print("samples shape:", samples.shape)
#     print("targets shape:", targets.shape)
#     break


# --------------------------------
# BiLSTM
# --------------------------------


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = torch.nn.LSTM(14, 16, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(16 * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, sequence_length, hidden_size * 2]
        lstm_out_last = lstm_out[:, -1, :]
        # shape: [batch_size, hidden_size * 2]
        out = self.fc(lstm_out_last)
        return out


net = MyModel().to(device)

print(net)
total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

num_epochs = 20
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

loss_v = np.empty(0)
loss_val_v = np.empty(0)
mae_v = np.empty(0)
mae_val_v = np.empty(0)

for epoch in range(num_epochs):
    train_loss = 0.0
    train_mae = 0.0
    val_mae = 0.0
    val_loss = 0.0
    batches_train = 0
    batches_val = 0

    net.train()
    for samples_train, targets_train in train_dataset:
        # print(samples.shape)
        targets_flat_train = targets_train.numpy().reshape(targets_train.shape[0], 1)
        torch_samples_train = torch.from_numpy(samples_train.numpy()).float().to(device)
        torch_targets_train = torch.from_numpy(targets_flat_train).float().to(device)
        optimizer.zero_grad()
        outputs_train = net(torch_samples_train)
        loss = criterion(outputs_train, torch_targets_train)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += np.mean(
            np.abs(
                targets_train.numpy().flatten()
                - outputs_train.detach().cpu().numpy().flatten()
            )
        )
        batches_train += 1

    net.eval()
    with torch.no_grad():
        for samples_val, targets_val in val_dataset:
            targets_flat_val = targets_val.numpy().reshape(targets_val.shape[0], 1)
            torch_samples_val = torch.from_numpy(samples_val.numpy()).float().to(device)
            torch_targets_val = torch.from_numpy(targets_flat_val).float().to(device)
            outputs_val = net(torch_samples_val)
            loss = criterion(outputs_val, torch_targets_val)
            val_loss += loss.item()
            val_mae += np.mean(
                np.abs(
                    targets_val.numpy().flatten()
                    - outputs_val.detach().cpu().numpy().flatten()
                )
            )
            batches_val += 1

    train_loss = train_loss / batches_train
    val_loss = val_loss / batches_val
    train_mae = train_mae / batches_train  # samples_seen_train
    val_mae = val_mae / batches_val  # samples_seen_val

    print(
        "Epoch {:02d}: loss {:.4f} - train mae {:.4f} - val. loss {:.4f} - val. mae {:.4f}".format(
            epoch + 1, train_loss, train_mae, val_loss, val_mae
        )
    )
    # print(batches_train, "-", batches_val)

    loss_v = np.append(loss_v, train_loss)
    loss_val_v = np.append(loss_val_v, val_loss)
    mae_v = np.append(mae_v, train_mae)
    mae_val_v = np.append(mae_val_v, val_mae)

import matplotlib.pyplot as plt

num_epochs_stop = len(loss_val_v)
epochs = range(1, num_epochs_stop + 1)

plt.figure()
plt.plot(epochs, loss_v, "b-o", label="Training ")
plt.plot(epochs, loss_val_v, "r-o", label="Validation ")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("09.BiLSTM.Loss.png")

plt.figure()
plt.plot(epochs, mae_v, "b-o", label="Training ")
plt.plot(epochs, mae_val_v, "r-o", label="Validation ")
plt.title("Training and validation MAE")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("09.BiLSTM.MAE.png")
