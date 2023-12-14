from torchvision import datasets
from torchvision.transforms import ToTensor

#Load data from the internet
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)

print(train_data)

print(test_data)
#Data loaders
from torch.utils.data import DataLoader

loaders = {
    'train' : DataLoader(train_data,
                         batch_size=100,
                         shuffle=True,
                         num_workers=1),
    'test' : DataLoader(test_data,
                         batch_size=100,
                         shuffle=True,
                         num_workers=1)
}
