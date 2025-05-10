from GAN_fashion import Classifier, Generator, Discriminator
import torch

Classifier = Classifier()
G = Generator()
D = Discriminator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
G.load_state_dict(torch.load("G.pth"))
D.load_state_dict(torch.load("D.pth"))
