import torch
import matplotlib.pyplot as plt
from model import VAE
from train import train_loader
from model import VAE
from visualize import test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(DEVICE)
model.load_state_dict(torch.load("vae.pth", map_location=DEVICE))
model.eval()

x, _ = next(iter(test_loader))
x1 = x[0].to(DEVICE)
x2 = x[1].to(DEVICE)

with torch.no_grad():
    mu1, _ = model.encoder(x1.view(1, -1))
    mu2, _ = model.encoder(x2.view(1, -1))

# Interpolation steps
steps = 8
alphas = torch.linspace(0, 1, steps)
interpolated = []

with torch.no_grad():
    for a in alphas:
        z = (1 - a) * mu1 + a * mu2 #slide between them
        img = model.decoder(z)
        interpolated.append(img)

fig, axes = plt.subplots(1, steps, figsize=(12, 2))
for i in range(steps):
    axes[i].imshow(
        interpolated[i].view(28, 28).cpu(),
        cmap="gray"
    )
    axes[i].axis("off")
plt.show()