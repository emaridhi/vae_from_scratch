# VARIATIONAL AUTOENCODER (VAEs) on MNIST
implementing a research paper from scratch

This project is a implementation of a Variational Autoencoder (VAE) from scratch based on the paper "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014). 
The goal is to reproduce the core experiments of the original paper using PyTorch and the MNIST handwritten digits dataset.

The project includes: 
* Building the VAE architecture 
* Training with the ELBO objective Reconstructing test images 
* Generating new samples 
* Visualizing latent space behavior

## Requirements:
* Python 3.8+
* PyTorch
* torchvision
* numpy
* matplotlib

Install dependencies <br>
`pip install torch torchvision matplotlib numpy`

## How to Run:
1. Train the Model: Trains the VAR and save the model as vae.pth <br>
`python train.py`

4. Visalize Reconstructions and Generation: Shows the original vs reconstructed images and randomly generates images <br>
`python visualize.py`

5. Latent Space Interpolation: Displays smooth transitions between two randomaly generated digits in latent space <br>
`python interpolate.py`

## Expected Results:
* Reconstruction: The reconstructed digits resemble the originals with a slight bluriness.
* Generation: Randomly generated handwritten digits in diverse styles.
* Interpolation: Smooth morphing between digits with no abrupt transitions.

## Reference
Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. International Conference on Learning Representations (ICLR)

