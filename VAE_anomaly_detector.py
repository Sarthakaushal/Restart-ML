# Implement a VAE and train it on the MNIST dataset. Use the trained model to detect anomalies
# by evaluating the reconstruction error. Test your model on a set of anomalous images (e.g., noisy
# or corrupted MNIST digits) and plot the distribution of reconstruction errors. Set a threshold to
# classify images as "anomalous" or "normal".

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from core.dataLoaders.mnist import MNIST

training = True
Anomaly_det = True

## load data loader same as VAE.py
transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 16
data_cls = MNIST(transforms=transform, batch_size=batch_size)

# create data loader
[train_loader, test_loader] = data_cls.create_loader()
train_data_sample_shape = data_cls.view_sample_stats()


# Model parameters
input_dim = 784  # 28x28 pixels flattened
hidden_dim = 400
latent_dim = 200
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
lr = 0.001

# # Initialize model
from core.models.VAE import VAE
model = VAE(input_dim, hidden_dim, latent_dim)
model = model.to(device)

# Loss function
def loss_function(x, x_hat, mean, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + KLD, reconstruction_loss, KLD

# # Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
if training:
    # train the model
    # Training parameters
    epochs = 51
    train_losses = []
    kld_losses = []
    reconstruction_losses = []

    print("Starting training...")
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
        
        # Calculate average losses for epoch
        avg_loss = overall_loss / (batch_idx * batch_size)
        avg_kld = epoch_kld / (batch_idx * batch_size)
        avg_recon = epoch_recon / (batch_idx * batch_size)
        
        train_losses.append(avg_loss)
        kld_losses.append(avg_kld)
        reconstruction_losses.append(avg_recon)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, KLD: {avg_kld:.4f}, Reconstruction: {avg_recon:.4f}")

    print("Training finished!")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'kld_losses': kld_losses,
        'reconstruction_losses': reconstruction_losses,
    }, 'model_dump/VAE/vae_anomaly_model.pth')

    print("Model saved successfully!")

#load model
if not training:
    checkpoint = torch.load('model_dump/VAE/vae_anomaly_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['train_losses']
    kld_losses = checkpoint['kld_losses'] 
    reconstruction_losses = checkpoint['reconstruction_losses']

    print("Model loaded successfully!")

if Anomaly_det:
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
    data_cls = MNIST(transforms=transform, batch_size=1)

    # create data loader
    [train_loader, test_loader] = data_cls.create_loader()
    # Anomaly Detection on Test Data
    model.eval()
    reconstruction_errors = []
    x_test = []
    with torch.no_grad():
        for x, _ in test_loader:
            # Add Gaussian noise with 10% probability
            if np.random.random() < 0.1:
                x = x + torch.randn_like(x)
            
            x = x + 0.1 * torch.randn_like(x)
            x = torch.clamp(x, min=0.0, max=1.0)
            x = x.view(-1, input_dim).to(device)
            x_hat, mean, log_var = model(x)
            recon_error = nn.functional.binary_cross_entropy(x_hat, x, reduction='none').sum(dim=1)
            reconstruction_errors.extend(recon_error.cpu().numpy())
            x_test.append(x)

    # Plot Reconstruction Error Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(reconstruction_errors, bins=50, edgecolor='black')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title("Distribution of Reconstruction Errors on Test Data")
    plt.savefig('data/vae_generated_images/vae_loss_distribution.png')
    plt.close()

    # Set threshold and classify anomalies
    threshold = np.percentile(reconstruction_errors, 95)
    anomalous_data_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

    print(f"Anomaly detection threshold set at {threshold:.4f}.")
    print(f"Number of anomalous samples detected: {len(anomalous_data_indices)}")

    # Optional: Visualize some anomalies
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, idx in enumerate(anomalous_data_indices[:9]):
        x = x_test[idx]
        print(x.shape)
        x = x.view(-1, input_dim).to(device)
        x_hat, _, _ = model(x)
        x = x.view(28, 28).cpu().numpy()
        x_hat = x_hat.view(28, 28).cpu().detach().numpy()
        ax = axes[i // 3, i % 3]
        ax.imshow(np.concatenate([x, x_hat], axis=1), cmap='gray')
        ax.axis('off')
    plt.suptitle("Anomalous Samples (Original | Reconstruction)")
    plt.show()