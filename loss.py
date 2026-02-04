import torch
import torch.nn.functional as F

def reconstruction_loss(x, x_hat): #measures how close the x hat is with x
    """
    x, x_hat: (batch_size, 784)
    """
    return F.binary_cross_entropy( #cuz we used a sigmoid activation function, we get [0,1]. goal is to 
        x_hat, x, reduction="sum"
    )

def kl_divergence(mu, logvar): #enforces N(0,1) for every latent dist
    """
    mu, logvar: (batch_size, latent_dim)
    """
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

def vae_loss(x, x_hat, mu, logvar):
    recon = reconstruction_loss(x, x_hat)
    kl = kl_divergence(mu, logvar)
    return recon + kl