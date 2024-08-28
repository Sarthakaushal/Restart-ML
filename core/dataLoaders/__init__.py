#core/dataLoaders/__init__.py
import os
class VisionDataTemplate():
    def __init__(self, 
                local_dir: str , 
                batch_size: int
                ) -> None:
        self.name = ''
        self.root_dir = None
        self.batch_size = None
    
    def _is_available_locally(self)->bool:
        """Checks if data is present in root_dir on local"""
        data_dir_contents = os.listdir(self.root_dir)
        if self.name in data_dir_contents:
            return True
        return False
    
    def _load_data(self):
        """Downloads data from source if not present in the root_dir on local"""
        pass
    
    def create_loader(self):
        """Creates sampler over the dataset and provides iterable over the 
        dataset"""
        pass
    
    def view_sample_stats(self):
        print('\n------------- Train Stats --------------\n')
        for X, y in self.train_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        data_sample_shape = X.shape
        print('\n------------- test Stats  --------------\n')
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        print('\n-------------    END      ---------------\n')
        return data_sample_shape