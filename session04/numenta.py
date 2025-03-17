import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeseries
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA


import AISclonalSelectionV4
import TWClonalSelV2
import TWClonalSelV3
import TWClonalSelV4
import TWNegSelectionV2


# Download latest version
path = kagglehub.dataset_download("boltzmannbrain/nab")

# We get the correct data
correctData = pd.read_csv(
    os.path.join(
        path, "artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv"
    )
)
correctData["anomaly"] = 0  # No anomalies

anomalyPath = os.path.join(path, "artificialWithAnomaly/artificialWithAnomaly/")


incorrectData = None

for file in os.listdir(anomalyPath):
    if file.startswith("art_daily"):
        df = pd.read_csv(os.path.join(anomalyPath, file))
        # We rename the second column
        name = file[len("art_daily_") : -len(".csv")]
        df.columns = ["timestamp", name]
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if incorrectData is None:
            incorrectData = df  # Initialize with the first file's DataFrame
        else:
            incorrectData = pd.merge(
                incorrectData, df, on="timestamp", how="outer"
            )  # Merge on timestamp

incorrectData = incorrectData.sort_values(by="timestamp")

# There is an error in flatmiddle.csv
# Until we are able to fix it, we will drop it
incorrectData = incorrectData.drop(columns="flatmiddle")


# The Anomalies are from the 2014-04-11 9:00 to 2014-04-11 18:55
# We mark it in a column
incorrectData["anomaly"] = 0

incorrectData.loc[
    (incorrectData["timestamp"] >= "2014-04-11 09:00:00")
    & (incorrectData["timestamp"] <= "2014-04-11 18:55:00"),
    "anomaly",
] = 1

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# We can plot the data
plt.figure(figsize=(12, 6))

for col in incorrectData.columns[1:]:

    if col == "anomaly":
        continue
    # plt.figure(figsize=(12, 6))

    plt.plot(incorrectData["timestamp"], incorrectData[col], label=col)

    plt.legend()

# plt.show()

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

X_TRAIN, _ = timeseries.from_data_to_timeseries(correctData)
# Y_Train doesn't exist because the models think there is always no anomalies.

X_TEST = Y_TEST = None

for col in incorrectData.columns[1:]:
    if col == "anomaly" or col == "timestamp":
        continue
    X, y = timeseries.from_data_to_timeseries(
        incorrectData.rename(columns={col: "value"})
    )

    if X_TEST is None:
        X_TEST = X
        Y_TEST = y
    else:
        X_TEST = np.concatenate((X_TEST, X))
        Y_TEST = np.concatenate((Y_TEST, y))

print(f"Number of windows in the training set: {X_TRAIN.shape[0]}")
print(f"Number of windows in the test set: {X_TEST.shape[0]}")

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################


pca = PCA(n_components=2, random_state=777)
pca.fit(X_TRAIN)

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
seed = 42
generations = 10
numIndividuals = 30
mutationStd = 0.9
numberOfClones = 20

models = [
    {
        "name": "Artificial Immune System Clonal Selection V4",
        "model": AISclonalSelectionV4.ClonalAISAnomaly(
            n_generations=generations,
            pop_size=numIndividuals,
            mutation_std=mutationStd,
            clone_factor=numberOfClones,
            random_seed=seed,
        ),
    },
    {
        "name": "Clonal Selection V2",
        "model": TWClonalSelV2.ClonalSelectionAIS(
            max_gens=generations,
            pop_size=numIndividuals,
            mutation_std=mutationStd,
            clone_factor=numberOfClones,
            random_seed=seed,
        ),
    },
    {
        "name": "Clonal Selection V3",
        "model": TWClonalSelV3.ClonalSelectionAIS(
            max_gens=generations,
            pop_size=numIndividuals,
            mutation_std=mutationStd,
            clone_factor=numberOfClones,
            random_seed=seed,
        ),
    },
    {
        "name": "Clonal Selection V4",
        "model": TWClonalSelV4.ClonalSelectionAIS(
            max_gens=generations,
            pop_size=numIndividuals,
            mutation_std=mutationStd,
            clone_factor=numberOfClones,
            random_seed=seed,
        ),
    },
    {
        "name": "Negative Selection V2",
        "model": TWNegSelectionV2.NegativeSelectionVectors(
            n_generations=generations, n_detectors=numIndividuals, random_seed=seed
        ),
    },
]


for model in models:
    model["model"].fit(X_TRAIN, pca_2d=pca)

    y_pred = model["model"].predict(X_TEST)

    model["cm"] = confusion_matrix(Y_TEST, y_pred)
    model["classification_report"] = classification_report(Y_TEST, y_pred)


for model in models:
    print(f"Model: {model['name']}")
    print(f"Confusion Matrix:")
    print(model["cm"])
