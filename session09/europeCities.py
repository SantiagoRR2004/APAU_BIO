import os
import kagglehub
import pandas as pd
import numpy as np


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
