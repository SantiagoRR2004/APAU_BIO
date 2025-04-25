import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random, sys
import matplotlib.pyplot as plt
import numpy as np


DATASET_DIR = "/home/leandro/datasets/img_align_celeba"
INPUT_DIM = (128,128,3)
# BATCH_SIZE = 512


# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print("GPU available:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device('cpu')

custom_transform = transforms.Compose([
    transforms.Resize(INPUT_DIM[:2]),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=custom_transform)
print("Length of dataset:", len(dataset), "images")
# dataset_loader = DataLoader(dataset, BATCH_SIZE, shuffle = True)


# # --------------------------------
# # Shows a random image
# # --------------------------------

# num = random.randrange(len(dataset))

# fig, ax = plt.subplots( nrows=1, ncols=1 )
# img_to_show = dataset[num][0].permute(1,2,0).numpy()
# ax.imshow(img_to_show)
# plt.tight_layout()
# plt.savefig("06.RandomFace.png")


# --------------------------------
# AE loading
# --------------------------------

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Encoder (3,128,128) -> (32,64,64) -> (64,32,32) -> (64,16,16) -> (64,8,8) -> 4096 -> 200
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(4096, 200),
        )
         
        # Decoder 200 -> 4096 -> (64,8,8) -> (64,16,16) -> (64,32,32) -> (32,64,64) -> (3,128,128)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(200, 4096),
            torch.nn.Unflatten(1,(64,8,8)),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
            
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
   
model_AE = AE()
model_AE.load_state_dict(torch.load('/home/leandro/models/AutoEncoders/06.AE_Faces.pth', weights_only=True))
model_AE.eval()


print("\nAE model loaded from file")
total_params = sum(p.numel() for p in model_AE.parameters())
print(f"Number of parameters in AE: {total_params}")


# --------------------------------
# VAE loading
# --------------------------------


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Encoder (3,128,128) -> (32,64,64) -> (64,32,32) -> (64,16,16) -> (64,8,8) -> 4096 -> 200
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            # torch.nn.Linear(4096, 200),
        )

        self.mean_layer = torch.nn.Linear(4096, 200)
        self.logvar_layer = torch.nn.Linear(4096, 200)
         
        # Decoder 200 -> 4096 -> (64,8,8) -> (64,16,16) -> (64,32,32) -> (32,64,64) -> (3,128,128)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(200, 4096),
            torch.nn.Unflatten(1,(64,8,8)),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
            
        )
 
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std*epsilon
        return z
    
    def decode(self, x):
        return self.decoder(x)
 
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


   
model_VAE = VAE()
model_VAE.load_state_dict(torch.load('/home/leandro/models/AutoEncoders/06.VAE_Faces.pth', weights_only=True))
model_VAE.eval()


print("\nVAE model loaded from file")
total_params = sum(p.numel() for p in model_VAE.parameters())
print(f"Number of parameters in VAE: {total_params}")


# --------------------------------
# AE: 10 random images
# --------------------------------

n_to_show = 10

input_idx = np.random.choice(range(len(dataset)), n_to_show)
input_images = torch.stack([ dataset[i][0] for i in input_idx ])

# --------------------------------

output_images = model_AE(input_images)

fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = input_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = output_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.AE.orig.png")

# --------------------------------
# AE: 10 noisy images
# --------------------------------

encoded_images = model_AE.encoder(input_images).detach().numpy()
encoded_images += np.random.normal(0.0, 1.0, size = (n_to_show,encoded_images.shape[1]))

output_images = model_AE.decoder(torch.tensor(encoded_images))

fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = input_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = output_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.AE.noisy.png")


# --------------------------------
# AE: 10 pure noise
# --------------------------------

encoded_noise = np.random.normal(0.0, 1.0, size = (n_to_show,encoded_images.shape[1]))
output_images = model_AE.decoder(torch.tensor(encoded_noise, dtype=torch.float32))
fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
for i in range(n_to_show):
    img = output_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(1, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.AE.purenoise.png")




# --------------------------------
# VAE: 10 random images
# --------------------------------


output_images, _, _ = model_VAE(input_images)

fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = input_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = output_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.VAE.orig.png")


# --------------------------------
# VAE: 10 noisy images
# --------------------------------

mean, logvar = model_VAE.encode(input_images)
encoded_images = model_VAE.reparameterization(mean, logvar).detach().numpy()
encoded_images += np.random.normal(0.0, 1.0, size = (n_to_show,encoded_images.shape[1]))
output_images = model_VAE.decoder(torch.tensor(encoded_images))

fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = input_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)
for i in range(n_to_show):
    #img = images[i].squeeze()
    img = output_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.VAE.noisy.png")


# --------------------------------
# VAE: pure noise
# --------------------------------

encoded_noise = np.random.normal(0.0, 1.0, size = (n_to_show,encoded_images.shape[1]))
output_images = model_VAE.decoder(torch.tensor(encoded_noise, dtype=torch.float32))
fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
for i in range(n_to_show):
    img = output_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(1, n_to_show, i+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.VAE.purenoise.png")



# --------------------------------
# VAE: morphing
# --------------------------------

input_idxs = np.random.choice(range(len(dataset)),2)
limit_images = torch.stack([ dataset[i][0] for i in input_idxs ])


fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)

image_left = limit_images[0].permute(1,2,0).numpy()
sub = fig.add_subplot(2, n_to_show, 1)
sub.axis('off')        
sub.imshow(image_left)

image_right = limit_images[1].permute(1,2,0).numpy()
sub = fig.add_subplot(2, n_to_show, n_to_show)
sub.axis('off')        
sub.imshow(image_right)

mean, logvar = model_VAE.encode(limit_images)
limit_encodings = model_VAE.reparameterization(mean, logvar).detach().numpy()
interp_encodings = np.zeros((n_to_show, limit_encodings.shape[1]))
interp_encodings[0] = limit_encodings[0]
interp_encodings[-1] = limit_encodings[-1]
step_size = 1 / (interp_encodings.shape[0] - 1)
for i in range(1, interp_encodings.shape[0] - 1):
    interpolation_factor = i * step_size
    interp_encodings[i] = (1 - interpolation_factor) * limit_encodings[0] + interpolation_factor * limit_encodings[-1]

interp_images = model_VAE.decoder(torch.tensor(interp_encodings, dtype=torch.float32))
for i in range(n_to_show):
    img = interp_images[i].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(2, n_to_show, n_to_show+i+1)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.VAE.morphing.png")



# --------------------------------
# VAE: distort image
# --------------------------------

input_idx = np.random.choice(range(len(dataset)))

input_image = dataset[input_idx][0]

fig = plt.figure(figsize=(10, 12))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
img = input_image.permute(1,2,0).numpy()
sub = fig.add_subplot(6, 5, 1)
sub.axis('off')        
sub.imshow(img)

output_image, _, _ = model_VAE(torch.stack([input_image]))
img = output_image[0].detach().permute(1,2,0).numpy()
sub = fig.add_subplot(6, 5, 5)
sub.axis('off')        
sub.imshow(img)

for i in range(0,20):
    mean, logvar = model_VAE.encode(torch.stack([input_image]))
    encoded_input2 = model_VAE.reparameterization(mean, logvar).detach().numpy()
    p = np.random.randint(0,200)
    encoded_input2[0][p] = encoded_input2[0][p] + 5
    output = model_VAE.decoder(torch.tensor(encoded_input2, dtype=torch.float32))
    img = output[0].detach().permute(1,2,0).numpy()
    sub = fig.add_subplot(6, 5 , i+6)
    sub.axis('off')        
    sub.imshow(img)

fig.savefig("08.VAE.distort.png")

