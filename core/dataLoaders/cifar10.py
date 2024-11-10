import torch
from ..dataLoaders import VisionDataTemplate
from torchvision import datasets
from torchvision.transforms import Compose ,ToTensor, Normalize
from torch.utils.data import DataLoader
class Conf:
    Root_data_dir = 'data'
    Batch_size = 32
    Transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class CIFAR10(VisionDataTemplate):
    def __init__(self, 
                local_dir: str = Conf.Root_data_dir,
                batch_size: int = Conf.Batch_size,
                transforms = ToTensor()
                ) -> None:
        self.name = 'CIFAR10'
        self.root_dir = local_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.labels = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    def _load_data(self):
        if not self._is_available_locally():
            download = True
        else:
            download = False
        
        training_data = datasets.CIFAR10( 
            root=self.root_dir,
            train=True,
            download=download,
            transform=self.transforms,
        )
        
        test_data = datasets.CIFAR10(
            root=self.root_dir,
            train=False,
            download=download,
            transform=self.transforms,
        )
        return [training_data, test_data]
    
    def create_loader(self, collate_fn=None):
        [train_data, test_data] = self._load_data()
        
        # Create data loaders.
        self.train_dataloader = DataLoader(
            train_data, 
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )
        
        return self.train_dataloader, self.test_dataloader

if __name__ == "__main__":
    cifar10 = CIFAR10()
    cifar10.create_loader()