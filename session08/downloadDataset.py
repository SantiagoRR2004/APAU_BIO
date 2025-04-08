# download_animals10_khub.py

import kagglehub
import shutil
import os


def download_and_extract_animals10():
    # Download dataset using kagglehub
    print("Downloading dataset with kagglehub...")
    path = kagglehub.dataset_download("alessiocorrado99/animals10")

    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(currentDirectory, "animals10")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Move all files and folders to the output directory
    for item in os.listdir(path):
        source = os.path.join(path, item)
        destination = os.path.join(output_dir, item)
        if os.path.isdir(source):
            shutil.move(source, destination)
        else:
            shutil.move(source, destination)

    print(f"Dataset extracted to: {output_dir}")


download_and_extract_animals10()
