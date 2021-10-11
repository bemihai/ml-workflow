import contextlib
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from data import CIFAR10

# TODO: replace with lightning DataModule
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.247, 0.2435, 0.2616)),
])

with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    cifar_train = CIFAR10('../data/',
                          train=True,
                          download=True,
                          transform=train_transform)
    cifar_val = CIFAR10('../data/',
                        train=False,
                        download=True,
                        transform=train_transform)

kwargs = {'num_workers': 12, 'pin_memory': True}
train_loader = DataLoader(cifar_train, batch_size=32, shuffle=True, drop_last=True, **kwargs)
val_loader = DataLoader(cifar_val, batch_size=32, shuffle=False, drop_last=True, **kwargs)
