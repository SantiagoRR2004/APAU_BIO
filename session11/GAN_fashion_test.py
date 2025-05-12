from GAN_fashion import GeneratedDataset
from generatorSimple import Generator
from classifier import Classifier
import torch
import os
from torch.utils.data import DataLoader

currentDirectory = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


G = Generator().to(device)

# Load the model
G.load_state_dict(torch.load(os.path.join(currentDirectory, "G.pth")))

num_samples = 5
noise = torch.randn(num_samples, 100).to(device)
fake_images = G(noise).detach().cpu()


fake_images = fake_images.view(-1, 1, 28, 28)
generated_dataset = GeneratedDataset(fake_images)
generated_loader = DataLoader(generated_dataset, batch_size=128, shuffle=False)

# load model Classifier
classifier = Classifier()
classifier.load_model(os.path.join(currentDirectory, "Classifier.pth"))
classifier.class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
classifier.predict(generated_loader)
