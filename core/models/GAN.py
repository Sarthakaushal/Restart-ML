import torch
from typing import List
from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from ..models import MlModel
import sys

class Generator(MlModel):
    def __init__(self, z_dim:int, hidden_dim:List[int], out_dim:int) -> None:
        super(Generator, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dim)):
            self.layers.append(
                nn.Linear(z_dim if i == 0 else hidden_dim[i-1], hidden_dim[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.layers.append(nn.Tanh())
        
    def forward(self, x):
        return self.layers(x)

class Discriminator(MlModel):
    def __init__(self, inp_dim:int, hidden_dim:List[int]) -> None:
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dim)):
            self.layers.append(nn.Linear(inp_dim if i == 0 else hidden_dim[i-1], hidden_dim[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim[-1], 1))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        return self.layers(x)

class ConvGenerator(MlModel):  # Change from nn.Module to MlModel
    def __init__(self, latent_dim):
        super(ConvGenerator, self).__init__()
        self.latent_dim = latent_dim

        # Create individual layers instead of Sequential
        self.initial = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, 
                                        stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, 
                                      stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)

        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, 
                                      stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(True)

        self.conv3 = nn.ConvTranspose2d(128, 3, kernel_size=4, 
                                      stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv3(x)
        x = self.tanh(x)
        return x

class ConvDiscriminator(MlModel):
    def __init__(self, 
                 inp_dim:List[int], #(batch_size, 3, 32, 32)
                 hidden_dim:List[int] # [8, 16]
            ) -> None:
        super(ConvDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Final: 1 x 1 x 1
        )
        
    def forward(self, x):
        return self.layers(x)

        
class GAN(nn.Module):
    def __init__(self, 
                z_dim: List[int],# (bs, 100,1,1)
                gen_hidden_dim:List[int],
                disc_hidden_dim:List[int],
                im_dim: List[int],
                device:str,
                generator:MlModel,
                discriminator:MlModel,
                ):
        super(GAN, self).__init__()
        
        self.gen = generator(z_dim[1]).to(device)
        print("********** GAN - generator **********\n", self.gen)
        self.device = device
        print("===>",im_dim, disc_hidden_dim)
        self.disc = discriminator(im_dim, disc_hidden_dim).to(device)
        print("********** GAN - discriminator **********\n", self.disc)
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Replace existing _init_weights with this
        def weights_init(m):
            classname = m.__class__.__name__
            # print("classname", classname)
            if classname.find('ConvGenerator') != -1:
                pass
            elif classname.find('ConvDiscriminator') != -1:
                pass
            elif classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.gen.apply(weights_init)
        self.disc.apply(weights_init)
                    
    def generator_step(self, z):
        """Generate fake samples and get discriminator predictions"""
        fake_samples = self.gen(z)
        # print("fake_samples.shape", fake_samples.shape, fake_samples.device)
        predictions = self.disc.forward(fake_samples.detach())
        return fake_samples, predictions
    
    def discriminator_step(self, real_samples, fake_samples):
        """Get discriminator predictions for both real and fake samples"""
        # print("1. real_samples.shape", real_samples.shape, real_samples.device)
        real_predictions = self.disc.forward(real_samples.to(self.device))
        
        # print("2. fake_samples.shape", fake_samples.shape, fake_samples.device)
        fake_predictions = self.disc.forward(fake_samples.detach())
        return real_predictions, fake_predictions