import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("boltzmannbrain/nab")

# We get the correct data
correctData = pd.read_csv(
    os.path.join(
        path, "artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv"
    )
)

anomalyPath = os.path.join(path, "artificialWithAnomaly/artificialWithAnomaly/")


data = None

for file in os.listdir(anomalyPath):
    if file.startswith("art_daily"):
        df = pd.read_csv(os.path.join(anomalyPath, file))
        # We rename the second column
        name = file[len("art_daily_") : -len(".csv")]
        df.columns = ["timestamp", name]
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if data is None:
            data = df  # Initialize with the first file's DataFrame
        else:
            data = pd.merge(data, df, on="timestamp", how="outer")  # Merge on timestamp

data = data.sort_values(by="timestamp")

# There is an error in flatmiddle.csv
# Until we are able to fix it, we will drop it
data = data.drop(columns="flatmiddle")


# We can plot the data
plt.figure(figsize=(12, 6))

for col in data.columns[1:]:
    # plt.figure(figsize=(12, 6))

    plt.plot(data["timestamp"], data[col], label=col)

    plt.legend()

plt.show()
