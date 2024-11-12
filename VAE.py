from core.models.VAE import VAE
import torch.nn as nn
import torch
import torchvision.transforms as T
from torch.optim import Adam
from core.dataLoaders.mnist import MNIST
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_reconstructed_images(model, test_loader, device, num_images=10, epoch=0):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))[0][:num_images]
        batch = batch.to(device)
        batch_flattened = batch.view(-1, input_dim)
        reconstructed, _, _ = model(batch_flattened)
        reconstructed = reconstructed.view(-1, 1, 28, 28)
        
        comparison = torch.cat([batch, reconstructed])
        plt.figure(figsize=(12, 4))
        plt.imshow(make_grid(comparison.cpu(), nrow=num_images).permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'data/vae_generated_images/anomaly_reconstructions_{epoch}.png')
        plt.close()
        print(f"Reconstructed images saved for epoch {epoch}")

def generate_images(model, device, num_images=10, epoch=0):
    model.eval()
    with torch.no_grad():
        # Sample from normal distribution
        z = torch.randn(num_images, latent_dim).to(device)
        # Generate images
        generated = model.decoder(z)
        generated = generated.view(-1, 1, 28, 28)
        
        plt.figure(figsize=(12, 4))
        plt.imshow(make_grid(generated.cpu(), nrow=num_images).permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'data/vae_generated_images/anomaly_generated_images_{epoch}.png')
        plt.close()
        print(f"Generated images saved for epoch {epoch}")
if __name__ == "__main__":
    
    
    ############################
    # Set Model Hyperparameters
    ############################
    input_dim = 784
    hidden_dim = 400
    latent_dim = 200
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    epochs = 51
    batch_size = 16
    ############################
    
    
    ## Data Loading
    mnist = MNIST()
    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize((0.5,), (0.5,)),
    ])

    data_cls = MNIST(transforms=transform, batch_size=batch_size)

    # create data loader
    [train_loader, test_loader] = data_cls.create_loader()
    train_data_sample_shape  = data_cls.view_sample_stats()
    
   
    # Initialize Model
    model = VAE(input_dim, hidden_dim, latent_dim)
    
    print(model)

    BCE_loss = torch.nn.BCELoss()

    # Training metrics storage
    train_losses = []
    kld_losses = []
    reconstruction_losses = []

    def loss_function(x, x_hat, mean, log_var):
        reconstruction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstruction_loss + KLD, reconstruction_loss, KLD

    optimizer = Adam(model.parameters(), lr=lr)
    
    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        epoch_kld = 0
        epoch_recon = 0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, input_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss, recon_loss, kld_loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            epoch_kld += kld_loss.item()
            epoch_recon += recon_loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Record metrics
        avg_loss = overall_loss / (batch_idx * batch_size)
        avg_kld = epoch_kld / (batch_idx * batch_size)
        avg_recon = epoch_recon / (batch_idx * batch_size)
        
        train_losses.append(avg_loss)
        kld_losses.append(avg_kld)
        reconstruction_losses.append(avg_recon)
        
        print(f"\tEpoch {epoch + 1} complete! Average Loss: {avg_loss:.4f}, KLD: {avg_kld:.4f}, Reconstruction: {avg_recon:.4f}")
        
        # Visualize results every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_reconstructed_images(model, test_loader, device, 10, epoch)
            generate_images(model, device, 10, epoch)

    print("Finish!!")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Total Loss')
    plt.plot(kld_losses, label='KL Divergence')
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('data/vae_generated_images/anomaly_vae_training_curves.png')
    plt.close()

