import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from core.dataLoaders.mnist import MNIST
from core.models.VAE import VAE
from sklearn.manifold import TSNE

# Model parameters
input_dim = 784  # 28x28 pixels flattened
hidden_dim = 400
latent_dim = 200
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
lr = 0.001

# Initialize and load trained model
model = VAE(input_dim, hidden_dim, latent_dim)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load('model_dump/VAE/vae_anomaly_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare data
transform = transforms.Compose([transforms.ToTensor()])
data_cls = MNIST(transforms=transform, batch_size=100)
[train_loader, test_loader] = data_cls.create_loader()

# Collect latent vectors and labels
latent_vectors = []
labels = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view(-1, input_dim).to(device)
        # Get the mean vector from the VAE encoder
        _, mean, _ = model(data)
        latent_vectors.append(mean.cpu().numpy())
        labels.append(target.numpy())

# Concatenate all batches
latent_vectors = np.concatenate(latent_vectors, axis=0)
labels = np.concatenate(labels, axis=0)

# Use t-SNE to reduce dimensionality to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE Visualization of VAE Latent Space\nColored by Digit Class')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Save the visualization
plt.savefig('data/vae_generated_images/latent_space_visualization.png')
plt.close()
