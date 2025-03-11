import kagglehub
import shutil
from pathlib import Path


def get_data():

    # Download latest version
    path = kagglehub.dataset_download("boltzmannbrain/nab")

    print("Path to dataset files:", path)
    carpeta = Path(__file__).parent
    print("Path to current directory:", carpeta)
    shutil.move(path, f"{carpeta}/data")


if __name__ == "__main__":
    get_data()
