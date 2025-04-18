import os
import kagglehub
import pandas as pd


folder = kagglehub.dataset_download("orvile/european-cities-weather-prediction-dataset")
fname = os.path.join(folder, "weather_prediction_dataset.csv")

df = pd.read_csv(fname, delimiter=",", encoding="utf-8")

# Drop the column DATE and MONTH
df = df.drop(columns=["DATE", "MONTH"])

city_names = [
    col.replace("_temp_mean", "") for col in df.columns if col.endswith("_temp_mean")
]

cities = {c: {} for c in city_names}

# Get the raw data
for city in city_names:
    cities[city]["raw_data"] = df.loc[:, df.columns.str.startswith("city")]
