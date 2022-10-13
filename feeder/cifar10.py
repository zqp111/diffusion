import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

batch_size = 64

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, split='train', image_size=32):
        trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        if split == 'train':
            super().__init__(root='./data', train=True, transform=trans, download=True)
        else:
            super().__init__(root='./data', train=False, transform=trans, download=True)


if __name__ == '__main__':
    da = CIFAR10('./data')
    print(len(da))
    dad = CIFAR10('./data', split='val')
    print(len(dad))
