from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from yacs.config import CfgNode

from callbacks.callback import LoggerCallback


def build_callbacks(cfg: CfgNode, logger):
    callbacks = []

    early_stopping = EarlyStopping(
        monitor=cfg.train.es.monitor,
        mode=cfg.train.es.mode,
        verbose=cfg.train.es.verbose,
        patience=cfg.train.es.patience)

    lr_logger = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        filename="epoch-{epoch}-step-{step}-loss-{loss:.2f}",
        monitor="val_loss",
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks.extend([checkpoint_callback, LoggerCallback(logger=logger), early_stopping, lr_logger])

    return callbacks
