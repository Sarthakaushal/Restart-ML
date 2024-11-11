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
        Normalize((0.5,), (0.5,))
        ])
class MNIST(VisionDataTemplate):
    def __init__(self, 
                local_dir: str = Conf.Root_data_dir, 
                transforms = ToTensor(),
                batch_size = Conf.Batch_size
                ) -> None:
        self.name = 'MNIST'
        self.root_dir = local_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.labels = [x for x in range(10)]
        self.label_desc = {x: str(x) for x in range(10)}
        
    def _load_data(self):
        if not self._is_available_locally():
            download = True
        else:
            download = False
        
        training_data = datasets.MNIST(
            root=self.root_dir,
            train=True,
            download=download,
            transform=self.transforms,
        )
        
        test_data = datasets.MNIST(
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