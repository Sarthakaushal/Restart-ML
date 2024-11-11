from core.dataLoaders.cifar10 import CIFAR10
from core.dataLoaders.mnist import MNIST
import torch, torchvision
from torch import nn
import torchvision.transforms as T
from core.models.GAN import GAN
import os, time, mlflow
from core.dataLoaders import VisionDataTemplate
import yaml, sys
from tqdm import tqdm
from core.models.GAN import Generator, Discriminator, ConvGenerator, ConvDiscriminator

def train_gan(gan_model:GAN,
              dataloader:VisionDataTemplate,
              **kwargs
              ):
    # Logger
    mlflow.set_experiment("GAN_Training")
    print(kwargs)
    output_dir = kwargs.get('output_dir')
    device = kwargs.get('device')
    g_optimizer = kwargs.get('g_optimizer')
    print(g_optimizer, output_dir)
    if "Adam" in g_optimizer:
        g_optimizer = torch.optim.Adam
    d_optimizer = kwargs.get('d_optimizer')
    if "Adam" in d_optimizer:
        d_optimizer = torch.optim.Adam
    g_learning_rate = kwargs.get('g_learning_rate')
    g_betas_min = kwargs.get('g_betas_min')
    g_betas_max = kwargs.get('g_betas_max')
    d_learning_rate = kwargs.get('d_learning_rate')
    d_betas_min = kwargs.get('d_betas_min')
    d_betas_max = kwargs.get('d_betas_max')
    disc_train_skip_freq = kwargs.get('disc_train_skip_freq')
    logging_freq = kwargs.get('logging_freq')
    num_epochs = kwargs.get('num_epochs')
    tracking = kwargs.get('tracking')
    z_dim = kwargs.get('z_dim')
    batch_size = kwargs.get('batch_size')
    z_dim = [batch_size] + z_dim
    output_dir_ = output_dir
    model_type = kwargs.get('model_type')
    
    with mlflow.start_run(log_system_metrics=True):
        criterion = nn.BCELoss()
        print("Training Started!! Device: ", device)
    
        # Optimizers
        g_optimizer = g_optimizer(
            gan_model.gen.parameters(),
            lr=g_learning_rate, 
            betas=(g_betas_min, g_betas_max)
        )
        d_optimizer = d_optimizer(
            gan_model.disc.parameters(),
            lr=d_learning_rate, 
            betas=(d_betas_min, d_betas_max)
        )
        
        # logging params
        if tracking:
            mlflow.log_params({
                "num_epochs": num_epochs,
                "z_dim": z_dim,
                "g_learning_rate": g_learning_rate,
                "g_betas": (g_betas_min, g_betas_max),
                "d_learning_rate": d_learning_rate,
                "d_betas": (d_betas_min, d_betas_max)
            })
        len_dataloader = len(dataloader)
        total_disc_loss = 0
        total_gen_loss = 0

        for epoch in range(num_epochs):
            disc_loss = 0
            gen_loss = 0
            for batch_idx, [real_data, _] in enumerate(tqdm(dataloader)):
                batch_size = real_data.shape[0]
                z_dim[0] = batch_size
                
                # Create labels
                real_labels = torch.ones((batch_size, 1), device=device)
                fake_labels = torch.zeros((batch_size, 1), device=device)
                
                # -----------------
                #  Train Generator
                # -----------------
                g_optimizer.zero_grad()
                
                z = torch.randn(z_dim).to(device)
                if model_type == 'simple_gan':
                    z = z.view(z.size()[0], -1)
                
                fake_samples = gan_model.gen(z)
                fake_pred = gan_model.disc(fake_samples)
                
                g_loss = criterion(fake_pred, real_labels)
                g_loss.backward()
                g_optimizer.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                if batch_idx % disc_train_skip_freq == 0:
                    d_optimizer.zero_grad()
                    
                    if model_type == 'simple_gan':
                        real_data = real_data.view(real_data.size(0), -1)
                    
                    # Real samples
                    real_pred = gan_model.disc(real_data.to(device))
                    d_loss_real = criterion(real_pred, real_labels)
                    
                    # Fake samples
                    fake_pred = gan_model.disc(fake_samples.detach())
                    d_loss_fake = criterion(fake_pred, fake_labels)
                    
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    d_loss.backward()
                    d_optimizer.step()
                
                # Update losses
                disc_loss += d_loss.item() if batch_idx % disc_train_skip_freq == 0 else 0
                gen_loss += g_loss.item()
            
            avg_ep_d_loss = disc_loss/(len_dataloader/disc_train_skip_freq)
            avg_ep_g_loss = gen_loss/len_dataloader
            
            total_disc_loss += avg_ep_d_loss
            total_gen_loss += avg_ep_g_loss

            if tracking:
                mlflow.log_metrics({
                        "discriminator_loss": avg_ep_d_loss,
                        "generator_loss": avg_ep_g_loss
                    }, step=epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Discriminator Loss: ",
                  f"{avg_ep_d_loss:.2f}, Generator Loss: {avg_ep_g_loss:.2f}")
            
            # Visualize generated images every 10 epochs
            if epoch % logging_freq == 0:
                # Generate sample images
                with torch.no_grad():
                    z_test = torch.randn(z_dim).to(device)
                    if model_type == 'simple_gan':
                        z_test = z_test.view(z_test.size()[0], -1)
                    fake_samples, _ = gan_model.generator_step(z_test)
                    samples = fake_samples
                    samples = samples.view(samples.size(0), 1, 28, 28)
                    # Create grid of images
                    grid = torchvision.utils.make_grid(samples, nrow=4, 
                                                       normalize=True)
                    # Create output directory if it doesn't exist
                    if output_dir_ == output_dir:
                        output_dir_ = f'{output_dir}/{int(time.time())}'
                        os.makedirs(output_dir_, exist_ok=True)
                    # Save the grid as an image
                    save_path = os.path.join(output_dir_,
                                             f'gen_epoch_{epoch}.png')
                    torchvision.utils.save_image(grid, save_path)
                    mlflow.log_artifact(save_path, f'gen_epoch_{epoch}.png')
        
        print(f"Total Discriminator Loss: {total_disc_loss/num_epochs:.2f}")
        print(f"Total Generator Loss: {total_gen_loss/num_epochs:.2f}")
        mlflow.log_metrics({
            "total_discriminator_loss": total_disc_loss/num_epochs,
            "total_generator_loss": total_gen_loss/num_epochs
        })
        
if __name__ == "__main__":
    
    torch.manual_seed(22)
    
     # load yaml config
    with open('configs/model.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    print(config)
    
    # Create and train the model
    batch_size = config.get('simple_gan').get('batch_size')
    z_dim = config.get('simple_gan').get('z_dim')
    z_dim = [batch_size] + z_dim
    print("z_dim", z_dim)
    
    ## Data Loading
    if config.get('simple_gan').get('model_type') == 'simple_gan':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if config.get('simple_gan').get('model_type') == 'simple_gan':
        data_cls = MNIST(transforms=transform, batch_size=batch_size)
    else:
        data_cls = CIFAR10(transforms=transform, batch_size=batch_size)

    # create data loader
    [train_loader, test_loader] = data_cls.create_loader()
    train_data_sample_shape  = data_cls.view_sample_stats()
    
    data_dim = train_data_sample_shape  # for CIFAR10 (32x32x3 = 3072)
    gen_hidden_dim=config.get('simple_gan').get('gen_hidden_dim')
    disc_hidden_dim=config.get('simple_gan').get('disc_hidden_dim')
    
    if config.get('simple_gan').get('model_type') == 'simple_gan':
        print("Simple GAN Config:")
        print(z_dim, gen_hidden_dim, disc_hidden_dim, data_dim)
        gan_model = GAN(z_dim,
                        gen_hidden_dim,
                        disc_hidden_dim,
                        data_dim,
                        device='mps',
                        generator=Generator,
                        discriminator=Discriminator,
                        model_type='simple_gan')
    elif config.get('simple_gan').get('model_type') == 'dcgan':
        print("DCGAN Config:")
        print(z_dim, gen_hidden_dim, disc_hidden_dim, data_dim)
        gan_model = GAN(z_dim,
                        gen_hidden_dim,
                        disc_hidden_dim,
                        data_dim,
                        device='mps',
                        generator=ConvGenerator,
                        discriminator=ConvDiscriminator,
                        model_type='dcgan')
    # print(gan_model.gen)
    train_gan(gan_model, train_loader, **config.get('simple_gan'))
    
    # save model
    output_dir = config.get('simple_gan').get('model_save_dir')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    torch.save(gan_model.gen.state_dict(), f'{output_dir}/gan_model.pt')
    torch.save(gan_model.disc.state_dict(), f'{output_dir}/gan_model_disc.pt')
    