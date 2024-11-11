# latent space interpolation
## 1. Generate a random latent space
## 2. Interpolate between two points in the latent space
## 3. Generate images from the interpolated latent space
## 4. Plot the generated images

import torch
import matplotlib.pyplot as plt
import torchvision
from core.models.GAN import Generator
# generate alpha values between 0 and 1
def generate_alpha_values(start, end, num_points):
    return torch.linspace(start, end, num_points)

# load the model
def load_model(model_path):
    return torch.load(model_path)

# generate images from the interpolated latent space
def generate_images(model, alpha, z1,z2):
    z = alpha*z1 + (1-alpha)*z2
    return model(z)

# plot all the images in a line with one line for each alpha value
def plot_images(images):
    # create a grid based on the number of images
    n_images = images.size(0)
    print("n_images", images.size())
    grid = torchvision.utils.make_grid(images, nrow=n_images)
    # Convert from (C,H,W) to (H,W,C) format and normalize for plotting
    grid = grid.permute(1, 2, 0)  # Rearrange dimensions
    grid = grid.cpu().numpy()
    plt.figure(figsize=(15, 3))
    plt.axis('off')
    plt.imshow(grid, cmap='gray')
    plt.savefig('data/interpolated_images/gan_latent_space_interpolation.png')
    plt.show()
        
def main(): 
    model = Generator(
        z_dim = 784,
        hidden_dim = [128, 256, 512, 1024],
        out_dim = 784
    )
    model.load_state_dict(torch.load('model_dump/simple_gans/gan_model.pt'))
    model.eval()
    alpha = generate_alpha_values(0, 1, 5)
    z1 = torch.randn([1, 28, 28])
    z2 = torch.randn([1, 28, 28])
    # generate images
    # model prediction in z1 and z2
    z1_pred = model(z1.flatten(start_dim=1))
    z2_pred = model(z2.flatten(start_dim=1))
    #save and plot the z_pred
    plt.imshow(z1_pred.view(28, 28).detach().numpy(), cmap='gray')
    plt.savefig('data/interpolated_images/z1_pred.png')
    plt.imshow(z2_pred.view(28, 28).detach().numpy(), cmap='gray')
    plt.savefig('data/interpolated_images/z2_pred.png')
    images = []
    for i in range(len(alpha)):
        images.append(generate_images(model, alpha[i], z1.flatten(start_dim=1), z2.flatten(start_dim=1)))
    images = torch.stack(images)
    images = images.view(images.size(0), 1, 28, 28)
    plot_images(images)
    # save the plot
    
if __name__ == '__main__':
    main()