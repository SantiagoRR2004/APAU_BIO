import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_dim = 180

# Define the transformation (you can modify it with your custom transform)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(p=0.2),
    transforms.Resize((img_dim,img_dim)),
    transforms.ToTensor()
])

# Load your image
img_path = '/home/leandro/datasets/plane-helicopter/passenger-plane/00000149.jpg'  # Replace with your image path
original_img = Image.open(img_path)

# Number of transformations/mosaic size (3x3 grid)
num_variations = 9

# Apply the transform multiple times and store the results
transformed_imgs = [transform(original_img) for _ in range(num_variations)]

# Convert the tensors to PIL images
transformed_imgs = [transforms.ToPILImage()(img) for img in transformed_imgs]

# Create a mosaic (grid) of the transformed images
def show_mosaic(imgs, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, img in enumerate(imgs):
        row, col = divmod(i, cols)
        axes[row, col].imshow(np.array(img))
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig("dataaugmentation.png")

# Show the mosaic
show_mosaic(transformed_imgs, rows=3, cols=3)