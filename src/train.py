import logging

import mlflow
import pytorch_lightning as pl

from config import *
from data.data_module import CIFARDataModule
from callbacks.factory import build_callbacks
from models.classifier import OsnetClassifier

logger = logging.getLogger("ML-WORKFLOW")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("CIFAR10")

    mlflow.pytorch.autolog(silent=True)
    logger.info("MLFlow: pytorch lightning auto-logging is turned on. ")

    with mlflow.start_run(run_name="cifar_training"):

        # load config file (default config)
        cfg = get_cfg()
        cfg.data.root = '../data/'
        cfg.data.save_dir = '../logs'

        # initialize data module and model
        model = OsnetClassifier(cfg)
        dm = CIFARDataModule(cfg)
        dm.prepare_data()
        dm.setup(stage="fit")

        callbacks = build_callbacks(cfg, logger)
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epoch,
            enable_checkpointing=True,
            callbacks=callbacks,
            default_root_dir=cfg.data.save_dir,
        )

        trainer.fit(model, dm)
