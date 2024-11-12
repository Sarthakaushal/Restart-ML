from core.models import MlModel
import torch
import torch.nn as nn

class Encoder(MlModel):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.FC_in = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.FC_next = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.FC_Mean = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.FC_Var = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        x = self.FC_in(x)
        x = self.leaky_relu(x)
        x = self.FC_next(x)
        x = self.leaky_relu(x)
        mean = self.FC_Mean(x)
        log_var = self.FC_Var(x)
        return mean, log_var

class Decoder(MlModel):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.FC_in = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.FC_next = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.FC_out = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.FC_in(x)
        x = self.leaky_relu(x)
        x = self.FC_next(x)
        x = self.leaky_relu(x)
        return torch.sigmoid(self.FC_out(x))


class VAE(MlModel):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, torch.exp(0.5*log_var))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, log_var