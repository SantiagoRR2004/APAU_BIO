from GAN_fashion import Classifier, Generator, Discriminator, GeneratedDataset
import torch
from torch.utils.data import DataLoader

Classifier = Classifier()
G = Generator()
D = Discriminator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
G.load_state_dict(torch.load("G.pth"))
D.load_state_dict(torch.load("D.pth"))

num_samples = 5
noise = torch.randn(num_samples, 100).to(device)
fake_images = G(noise).detach().cpu()


fake_images = fake_images.view(-1, 1, 28, 28)
generated_dataset = GeneratedDataset(fake_images)
generated_loader = DataLoader(generated_dataset, batch_size=128, shuffle=False)

# load model Classifier
classifier = Classifier()
classifier.load_model("Classifier.pth")
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
