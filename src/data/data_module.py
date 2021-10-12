from torch.utils.data import DataLoader, random_split, IterableDataset
from torchvision import transforms
import pytorch_lightning as pl

from config import get_cfg
from data import CIFAR10


class CIFARDataModule(pl.LightningDataModule):
    """CIFAR10 Data Module"""
    
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            self.cfg = get_cfg()  # get default config if not provided
        else:
            self.cfg = cfg
        # define custom transforms for CIFAR10 dataset
        self.cfg.data.norm_mean = (0.4915, 0.4823, 0.4468)
        self.cfg.data.norm_std = (0.247, 0.2435, 0.2616)
        self.cfg.data.transforms = [transforms.ToTensor(),
                                    transforms.Normalize(self.cfg.data.norm_mean, self.cfg.data.norm_std)]
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.transform = transforms.Compose(self.cfg.data.transforms)
    
    def setup(self, stage=None):
        """Downloads the data, parse it and split the data into train, test, validation data."""
        self.df_train = CIFAR10(self.cfg.data.root,
                                train=True,
                                download=True,
                                transform=self.transform)
        val_size = int(self.cfg.data.split * len(self.df_train))
        train_size = len(self.df_train) - val_size
        self.df_train, self.df_val = random_split(self.df_train, [train_size, val_size])
        self.df_test = CIFAR10(self.cfg.data.root,
                               train=False,
                               download=True,
                               transform=self.transform)
    
    def _dataloader(self, df, batch_size, shuffle=False, **kwargs):
        """Generic function to create a torch DataLoader."""
        shuffle &= not isinstance(df, IterableDataset)
        return DataLoader(df, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          num_workers=self.cfg.data.num_workers, pin_memory=self.cfg.data.pin_memory, **kwargs)
    
    def train_dataloader(self):
        """Train data loader for the given input."""
        return self._dataloader(self.df_train, batch_size=self.cfg.train.batch_size)
    
    def val_dataloader(self):
        """Validation data loader for the given input."""
        return self._dataloader(self.df_val, batch_size=self.cfg.train.batch_size)
    
    def test_dataloader(self):
        """Test data loader for the given input."""
        return self._dataloader(self.df_test, batch_size=self.cfg.test.batch_size)
