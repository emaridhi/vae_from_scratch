import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(DEVICE)
model.load_state_dict(torch.load("vae.pth", map_location=DEVICE)) #loads the trained weights
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True
)

#RECONSTRUCTS IMAGES:
x, _ = next(iter(test_loader)) #gets one batch of images, labels
x = x.to(DEVICE)

with torch.no_grad(): #reconstructs them. switches of math. only generating here
    x_hat, _, _ = model(x) #full VAE pipeline. we get x hat which are the reconstructed images


fig, axes = plt.subplots(2, 8, figsize=(12, 4))

for i in range(8):
    axes[0, i].imshow( #Original
        x[i].view(28, 28).cpu(),
        cmap="gray"
    )
    axes[0, i].axis("off")

    axes[1, i].imshow( #Reconstructed
        x_hat[i].view(28, 28).cpu(),
        cmap="gray"
    )
    axes[1, i].axis("off")
plt.show()


#GENERATES NEW IMAGES: (random noise-> decoder-> new digit image)
z = torch.randn(8, 20).to(DEVICE) #8 random points in latent space

with torch.no_grad(): #just generating, no learning
    samples = model.decoder(z) #forms fake MNIST digits

fig, axes = plt.subplots(1, 8, figsize=(12, 2))

for i in range(8):
    axes[i].imshow(
        samples[i].view(28, 28).cpu(),
        cmap="gray"
    )
    axes[i].axis("off")
plt.show()