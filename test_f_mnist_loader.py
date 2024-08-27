from core.dataLoaders.fashion_mnist import Fashion_MNIST

#instantiate the data class
data_cls = Fashion_MNIST()

# create data loader
[train_data, test_data] = data_cls.create_loader()

data_cls.view_sample_stats()