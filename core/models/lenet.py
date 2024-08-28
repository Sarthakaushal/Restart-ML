#core/models/lenet.py
# [Source] : http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
import torch
from torch import nn
import torch.nn.functional as F
from core.models import MlModel

class LeNet(MlModel):
    def __init__(self, num_channels, num_classes) -> None:
        super(LeNet, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(num_channels, 6,(5,5), padding=2)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        # Second block
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        #Third Block
        self.conv3 = nn.Conv2d(16, 120, (5,5))
        # Fully Connected Layer 
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        # Output Layer
        self.output = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.avg_pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.avg_pool2(x)
        x = F.sigmoid(self.conv3(x))
        _, n_c, h, w = x.shape
        x = x.view(-1,n_c*h*w)
        x = F.sigmoid(self.fc1(x))
        x = self.output(x)
        return x