#core/models/fvsbn.py
# [Source] : http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

from core.models import MlModel
import torch
import torch.nn as nn
import torch.distributions as dist

class FVSBN(MlModel):
    def __init__(self,input_size, device) -> None:
        super(FVSBN, self).__init__()
        self.input_size = input_size
        
        self.weights = nn.Parameter(
            torch.rand(input_size, input_size, device=device)).tril()
        self.bias = nn.Parameter(torch.zeros(input_size, device=device))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, verbose=0):
        
        output = torch.zeros_like(x) # output dims same as input
        output[:,0] = self.sigmoid(self.bias[0]) 

        # Since weights is a lower triangular matrix, to match dims with the 
        # input we do W x X.T and then we do a transpose of the entire matrix 
        # to match dims with bias
        matmul = torch.matmul(
                self.weights, x.T, 
            ).T
        weighted_sum = self.sigmoid( matmul+ self.bias)
        output[:,1:] = weighted_sum[:,:-1] # 
        log_bernoulli_means = torch.log(output)
        log_likelihoods = x * (log_bernoulli_means) + (1 - x) * (
                                                        1 - log_bernoulli_means)
        return torch.sum(log_likelihoods)
    
    # Referenced from : https://github.com/ameya98/DeepGenerativeModels/blob/master/deepgenmodels/autoregressive/fvsbn.py
    def sample(self, num_samples):
        x = torch.randn(num_samples, self.input_size)
        bernoulli_mean_dim = torch.sigmoid(x.matmul(self.weights)+self.bias)
        bernoulli_mean_dim = bernoulli_mean_dim.nan_to_num(0)
        distribution = dist.bernoulli.Bernoulli(probs=bernoulli_mean_dim)
        x = distribution.sample()
        return torch.Tensor(x)
    
    
        
    def to_device(self, device):
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)
        
    