import torch
import os
import kagglehub
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")


folder = kagglehub.dataset_download("orvile/european-cities-weather-prediction-dataset")
fname = os.path.join(folder, "weather_prediction_dataset.csv")

df = pd.read_csv(fname, delimiter=",", encoding="utf-8")

# Drop the column DATE and MONTH
df = df.drop(columns=["DATE", "MONTH"])

# Turn the rest into numberic values
df = df.apply(pd.to_numeric, errors="coerce")

city_names = [
    col.replace("_temp_mean", "") for col in df.columns if col.endswith("_temp_mean")
]

cities = {c: {} for c in city_names}

num_train_samples = int(0.75 * len(df))
num_val_samples = len(df) - num_train_samples

print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)


# Get the data
for city in city_names:
    # Raw data
    data = df.loc[:, df.columns.str.startswith(city)]
    cities[city]["raw_data"] = data

    # Normalize the data (this normalizes all using only the training data)
    mean = data[:num_train_samples].mean(axis=0)
    data -= mean
    std = data[:num_train_samples].std(axis=0)
    data /= std

    # Temperature data
    cities[city]["temperature"] = df[f"{city}_temp_mean"].values

    # Make batches of 5 days
    windowsT = np.lib.stride_tricks.sliding_window_view(
        data[:num_train_samples], (5, data[:num_train_samples].shape[1])
    ).squeeze(axis=1)
    windowsV = np.lib.stride_tricks.sliding_window_view(
        data[num_train_samples:], (5, data[num_train_samples:].shape[1])
    ).squeeze(axis=1)

    targetsT = cities[city]["temperature"][5 : num_train_samples + 1]
    targetsV = cities[city]["temperature"][num_train_samples + 4 :]

    cities[city]["trainX"] = windowsT
    cities[city]["trainY"] = targetsT
    cities[city]["valX"] = windowsV
    cities[city]["valY"] = targetsV


# --------------------------------
# RNN
# --------------------------------


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = torch.nn.RNN(9, 16, batch_first=True)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)  # h_n has shape (1, batch_size, hidden_size)
        h_n = h_n.squeeze(0)  # Now shape is (batch_size, hidden_size)
        out = self.fc(h_n)  # Output shape is (batch_size, 1)
        return out


# Usage
net = MyModel().to(device)


num_epochs = 20
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

loss_v = np.empty(0)
loss_val_v = np.empty(0)
mae_v = np.empty(0)
mae_val_v = np.empty(0)

for city in city_names:

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_mae = 0.0
        val_mae = 0.0
        val_loss = 0.0
        batches_train = 0
        batches_val = 0

        net.train()
        for samples_train, targets_train in zip(
            cities[city]["trainX"],
            cities[city]["trainY"],
        ):
            targets_flat_train = np.reshape(targets_train, (-1, 1))
            torch_samples_train = torch.from_numpy(samples_train).float().to(device)
            torch_targets_train = (
                torch.from_numpy(targets_flat_train).float().to(device)
            )
            optimizer.zero_grad()
            outputs_train = net(torch_samples_train)
            loss = criterion(outputs_train, torch_targets_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mae += np.mean(
                np.abs(
                    targets_train.flatten()
                    - outputs_train.detach().cpu().numpy().flatten()
                )
            )
            batches_train += 1

        net.eval()
        with torch.no_grad():
            for samples_val, targets_val in zip(
                cities[city]["valX"],
                cities[city]["valY"],
            ):
                targets_flat_val = np.reshape(targets_val, (-1, 1))
                torch_samples_val = torch.from_numpy(samples_val).float().to(device)
                torch_targets_val = (
                    torch.from_numpy(targets_flat_val).float().to(device)
                )
                outputs_val = net(torch_samples_val)
                loss = criterion(outputs_val, torch_targets_val)
                val_loss += loss.item()
                val_mae += np.mean(
                    np.abs(
                        targets_val.flatten()
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

        loss_v = np.append(loss_v, train_loss)
        loss_val_v = np.append(loss_val_v, val_loss)
        mae_v = np.append(mae_v, train_mae)
        mae_val_v = np.append(mae_val_v, val_mae)
