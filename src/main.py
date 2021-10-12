from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from config import *
from data.data_module import CIFARDataModule
from metrics.accuracy import AccumulatedAccuracy
from models.classifier import OsnetClassifier

if __name__ == '__main__':
    
    # get CLI args
    parser = ArgumentParser(description="ML Workflow")
    parser.add_argument("--run-name", type=str, default="", help="Run name")
    parser.add_argument("--experiment-id", type=int, default=1, help="Experiment id")
    parser.add_argument("--tracking-uri", type=str, default="http://localhost:5000/", help="Tracking server URI")
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    dict_args = vars(args)
    
    # mlflow autologs to localhost:5000 by default
    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri(dict_args['tracking_uri'])
    
    # load config file (default config)
    cfg = get_cfg()
    cfg.data.root = '../data/'
    cfg.model.feature_dim = 64
    cfg.test.metrics = [AccumulatedAccuracy()]
    
    # initialize data module and model
    model = OsnetClassifier(cfg)
    dm = CIFARDataModule(cfg)
    dm.prepare_data()
    dm.setup(stage="fit")
    
    # define training callbacks
    early_stopping = EarlyStopping(monitor=cfg.train.es.monitor, mode=cfg.train.es.mode,
                                   verbose=cfg.train.es.verbose, patience=cfg.train.es.patience)
    lr_logger = LearningRateMonitor()
    
    # define trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=cfg.train.max_epoch,
        callbacks=[lr_logger, early_stopping],
        progress_bar_refresh_rate=100,
        default_root_dir=cfg.data.save_dir,
    )
    
    with mlflow.start_run(run_name=dict_args['run_name'], experiment_id=str(dict_args['experiment_id'])) as run:
        trainer.fit(model, dm)
        trainer.test()
