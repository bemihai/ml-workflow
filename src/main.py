import logging
from dotenv import load_dotenv

import mlflow
import pytorch_lightning as pl

from config import *
from data.data_module import CIFARDataModule
from callbacks.factory import build_callbacks
from metrics.accuracy import AccumulatedAccuracy
from models.classifier import OsnetClassifier

logger = logging.getLogger("ML-WORKFLOW")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    load_dotenv()

    mlflow.pytorch.autolog(silent=True)
    logger.info("MLFlow: pytorch lightning auto-logging is turned on. ")

    logger.info(f"MLFlow: tracking server : {mlflow.get_tracking_uri()}")
    logger.info(f"MLFlow: artifacts store : {mlflow.get_artifact_uri()}")

    # load config file (default config)
    cfg = get_cfg()
    cfg.data.root = '../data/'
    cfg.data.save_dir = '../logs'
    cfg.test.metrics = [AccumulatedAccuracy()]

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

    mlflow.end_run()
    with mlflow.start_run(run_name="cifar_class") as _:
        trainer.fit(model, dm)
        trainer.test()
