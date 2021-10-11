import contextlib
import os

import mlflow
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.backends import cudnn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_cfg
from data import CIFAR10
from losses.loss import CrossEntropyLoss
from metrics.accuracy import AccumulatedAccuracy
from models.networks import ClassificationNet
from models.osnet import OSNet, OSBlock
from optim import build_optimizer
from optim.scheduler import build_lr_scheduler


class OsnetClassifier(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.cfg = get_cfg()
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg.model.feature_dim = 64

        self.feature_extractor = OSNet(
            blocks=[OSBlock, OSBlock, OSBlock],
            layers=[1, 1, 1],
            channels=[16, 64, 96, 128],
            feature_dim=self.cfg.model.feature_dim
        )
        
        self.model = ClassificationNet(
            self.feature_extractor,
            feature_dim=self.cfg.model.feature_dim,
            n_classes=10
        )
        
        self.loss_fn = CrossEntropyLoss(
            num_classes=10,
            use_gpu=self.cfg.use_gpu,
            label_smooth=self.cfg.loss.softmax.label_smooth
        )
        
        self.optimizer = None
        self.scheduler = None
        self.metric = AccumulatedAccuracy()
        
        if self.cfg.use_gpu:
            cudnn.benchmark = True
            self.model = nn.DataParallel(self.model).cuda()
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        self.optimizer = build_optimizer(
            self.model,
            optim=self.cfg.train.optim,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay
        )
        lr_scheduler = build_lr_scheduler(
            self.optimizer,
            lr_scheduler=self.cfg.train.lr_scheduler,
            stepsize=self.cfg.train.stepsize,
            gamma=self.cfg.train.gamma
        )
        self.scheduler = {'scheduler': lr_scheduler, 'monitor': 'val_loss'}
        return [self.optimizer], [self.scheduler]
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x, y = self._parse_inputs(x, y)
        y_hat = self.model(*x)
        loss = self._compute_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.metric(y_hat, y), prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x, y = self._parse_inputs(x, y)
        y_hat = self.model(*x)
        loss = self._compute_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.metric(y_hat, y), prog_bar=True)
    
    def _compute_loss(self, outputs, target):
        loss_inputs = outputs
        if target:
            loss_inputs += target
        loss_outputs = self.loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        return loss
    
    def _parse_inputs(self, x, y):
        y = (y.to(self.device),) if len(y) > 0 else None
        if type(x) not in (tuple, list):
            x = (x,)
        x = tuple(d.to(self.device) for d in x)
        return x, y


def main():
    
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

    mlflow.pytorch.autolog()

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs', save_top_k=1, verbose=True, monitor="val_loss", mode="min")
    
    model = OsnetClassifier()
    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=500,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test()
    
    
if __name__ == '__main__':
    main()
