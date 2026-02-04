import torch
import torch.nn as nn
import torch.optim as optim #optimization algorithms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VAE
from loss import vae_loss

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([ #pipeline of transformations
    transforms.ToTensor(), #image -> pizel values [0,1]
    transforms.Lambda(lambda x: x.view(-1)) #flattens 28x28 -> 784. reshaping from matrices to vectors
])

train_dataset = datasets.MNIST( #transforms each MNIST image
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader( #makes batches of images to feed the model
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR) #decides how much to change the weight. Adam is an optimization strat: p + adaptive lr.


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (x, _) in enumerate(train_loader):

        x = x.to(DEVICE) #sends data to GPU/CPU
        optimizer.zero_grad() #erases the old gradients
        x_hat, mu, logvar = model(x) #forward pass
        loss = vae_loss(x, x_hat, mu, logvar)
        loss.backward() #backprop
        optimizer.step() #adam updates weights (gard descent)
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "vae.pth")
print("Model saved as vae.pth")