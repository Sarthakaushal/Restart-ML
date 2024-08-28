#core/dataLoaders/fashion_mnist.py
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import FashionMNIST 
from torchvision import datasets
from torchvision.transforms import ToTensor
from ..dataLoaders import VisionDataTemplate
import os

class Conf:
    Root_data_dir = 'data'
    Batch_size = 64

class Fashion_MNIST(VisionDataTemplate):
    def __init__(self, 
                local_dir: str = Conf.Root_data_dir, 
                transforms = ToTensor()
                ) -> None:
        self.name = 'FashionMNIST'
        self.root_dir = local_dir
        self.transforms = transforms
        self.batch_size = Conf.Batch_size
        self.labels = [x for x in range(10)]
        self.label_desc = {
            0 :'T-shirt/top', 1: 'Trouser', 2: "Pullover", 3: "Dress", 
            4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 
            9: "Ankle boot"
        }
    
    def _load_data(self):
        """Downloads data from source if not present in the root_dir on local"""
        if not self._is_available_locally():
            download =True
        else:
            download = False
        
        training_data = datasets.FashionMNIST(
            root=self.root_dir,
            train=True,
            download=download,
            transform=self.transforms,
        )
        
        test_data = datasets.FashionMNIST(
            root=self.root_dir,
            train=False,
            download=download,
            transform=self.transforms,
        )
        return [training_data, test_data]
    
    def create_loader(self):
        """Creates sampler over the dataset and provides iterable over the 
        dataset"""
        [train_data, test_data] = self._load_data()
        
        # Create data loaders.
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size)
        
        return self.train_dataloader, self.test_dataloader
        