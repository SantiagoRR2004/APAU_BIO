import kagglehub
import shutil
from pathlib import Path
import pandas as pd


def download_data():

    # Download latest version
    path = kagglehub.dataset_download("boltzmannbrain/nab")

    print("Path to dataset files:", path)
    carpeta = Path(__file__).parent
    print("Path to current directory:", carpeta)
    shutil.move(path, f"{carpeta}/data")


def get_data(normal="art_daily_small_noise.csv", anomaly="art_daily_jumpsup.csv"):
    path = Path(__file__).parent
    if not path.joinpath("data").exists():
        print("Data not found, downloading...")
        download_data()

    # Load normal data
    normalData = pd.read_csv(
        path.joinpath("data/artificialNoAnomaly/artificialNoAnomaly/" + normal)
    )

    # Load anomaly data
    anomalyData = pd.read_csv(
        path.joinpath("data/artificialWithAnomaly/artificialWithAnomaly/" + anomaly)
    )
    return normalData, anomalyData


if __name__ == "__main__":
    download_data()
