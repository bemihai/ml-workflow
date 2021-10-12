import numpy as np
import torch

from config import get_cfg
import pytorch_lightning as pl

from losses.loss import CrossEntropyLoss
from models.networks import ClassificationNet
from models.osnet import OSNet, OSBlock
from optim import build_optimizer
from optim.scheduler import build_lr_scheduler


class OsnetClassifier(pl.LightningModule):

    def __init__(self, cfg=None):
        """Initialize model, loss function, optimizer, lr scheduler from config file."""
        super().__init__()
        if cfg is None:
            self.cfg = get_cfg()  # get default config if not provided
        else:
            self.cfg = cfg
        self.optimizer = None
        self.scheduler = None
        self.feature_extractor = OSNet(
            blocks=[OSBlock, OSBlock],
            layers=[1, 1],
            channels=[16, 32, 64],
            feature_dim=self.cfg.model.feature_dim,
        )
        self.model = ClassificationNet(
            self.feature_extractor,
            feature_dim=self.cfg.model.feature_dim,
            n_classes=10
        )
        self.loss_fn = CrossEntropyLoss(
            num_classes=10,
            use_gpu=self.cfg.model.use_gpu,
            label_smooth=self.cfg.loss.softmax.label_smooth
        )
        
    def forward(self, x):
        """Forward step."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizer and lr scheduler."""
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
        """Training step: returns batch training loss and metrics."""
        x, y = self.parse_inputs(*train_batch)
        outputs = {}
        y_hat = self.model(*x)
        outputs['loss'] = self.compute_loss(y_hat, y)
        for metric in self.cfg.test.metrics:
            outputs[metric.name] = metric(y_hat, y)
        return outputs
    
    def training_epoch_end(self, outputs):
        """Computes epoch average loss and  metrics for logging."""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)
        for metric in self.cfg.test.metrics:
            avg_metric = float(np.mean([x[metric.name] for x in outputs]))
            self.log(f'train_{metric.name}', avg_metric)
            
    def validation_step(self, val_batch, batch_idx):
        """Validation step: returns batch validation loss and metrics."""
        x, y = self.parse_inputs(*val_batch)
        outputs = {}
        y_hat = self.model(*x)
        outputs['val_loss'] = self.compute_loss(y_hat, y)
        for metric in self.cfg.test.metrics:
            outputs[metric.name] = metric(y_hat, y)
        return outputs
    
    def validation_epoch_end(self, outputs):
        """Computes epoch average validation loss and metrics for logging."""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        for metric in self.cfg.test.metrics:
            avg_metric = float(np.mean([x[metric.name] for x in outputs]))
            self.log(f'val_{metric.name}', avg_metric)
        
    def test_step(self, test_batch, batch_idx):
        """Test step: returns model metrics."""
        x, y = self.parse_inputs(*test_batch)
        outputs = {}
        y_hat = self.model(*x)
        for metric in self.cfg.test.metrics:
            outputs[metric.name] = metric(y_hat, y)
        return outputs
    
    def test_epoch_end(self, outputs):
        """Computes epoch average metrics for logging."""
        for metric in self.cfg.test.metrics:
            avg_metric = float(np.mean([x[metric.name] for x in outputs]))
            self.log(f'test_{metric.name}', avg_metric)
    
    def compute_loss(self, outputs, target):
        """Computes the loss."""
        loss_inputs = outputs
        if target:
            loss_inputs += target
        loss_outputs = self.loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        return loss
    
    def parse_inputs(self, x, y):
        """Prepares inputs."""
        y = (y.to(self.device.type),) if len(y) > 0 else None
        if type(x) not in (tuple, list):
            x = (x,)
        x = tuple(d.to(self.device.type) for d in x)
        return x, y
