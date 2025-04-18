import os
import kagglehub
import pandas as pd
import csv


folder = kagglehub.dataset_download("saurabhshahane/meteorological-data-set")

islands = {}

for file in os.listdir(folder):
    if file.endswith(".csv"):
        key = file.replace("Dataset_", "").replace(".csv", "")
        path = os.path.join(folder, file)
        islands[key] = {"filePath": path}


for key, value in list(islands.items()):
    with open(value["filePath"], "r", encoding="utf-8") as f:
        # Read a sample to guess the delimiter
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter

    df = pd.read_csv(value["filePath"], delimiter=delimiter, encoding="utf-8")

    # Remove the " symbol from data
    df = df.replace('"', "", regex=True)

    # Drop 'id' and any columns that start with 'Unnamed'
    df = df.drop(
        columns=[col for col in df.columns if col == "id" or col.startswith("Unnamed")]
    )

    # Drop columns that have NA
    df = df.dropna(axis=1)
    # This is could be fixed, but now we are doing the most simple thing

    # Delete from the dictionary if only one column remains
    if len(df.columns) == 1:
        del islands[key]

print("Las islas válidas son:")
for island in islands.keys():
    print(island)
