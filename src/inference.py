import argparse

import mlflow
import pytorch_lightning as pl

from src.callbacks.factory import build_callbacks
from src.config import get_cfg
from src.data.data_module import CIFARDataModule
from src.metrics.accuracy import AccumulatedAccuracy
from src.models.classifier import OsnetClassifier
from src.train import logger

mlflow.tracking.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CIFAR10")
mlflow_artifacts_path = "/Users/lu80mf/personal/ml-workflow/mlartifacts/"


def get_model_from_registry(model_name: str, alias: str | None = None) -> dict[str, str]:
    """Gets a trained model version from the registry. Returns the run_id and the best checkpoint path."""
    # look up model_name in the registry
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models(f"name='{model_name}'").to_list()
    if len(models) == 0:
        raise ValueError(f"There is no model registered with name: '{model_name}'")
    if len(models) > 1:
        raise ValueError(f"There are multiple models registered with name: '{model_name}'")
    model = models[0]

    # get the latest model version if alias is not specified
    if not alias:
        versions = client.search_model_versions(f"name='{model_name}'", order_by=["creation_timestamp DESC"])
        model_version = versions[0]
    else:
        try:
            model_version = client.get_model_version_by_alias(model.name, alias)
        except mlflow.exceptions.MlflowException as e:
            if e.message == f"INVALID_PARAMETER_VALUE: Registered model alias {alias} not found.":
                raise ValueError(f"Model alias {alias} not found.") from e
            raise e

    # get the run associated to this model version and return the run_id and the best checkpoint path
    mv_run = client.get_run(model_version.run_id)
    # if "best_ckpt" not in mv_run.data.params:
    #     raise  ValueError(f"Model version '{model_version.version}' does not have 'best_ckpt' parameter.' "
    #                       f"Training run id: {mv_run.info.run_id}")

    artifacts_uri = mv_run.info.artifact_uri.replace("mlflow-artifacts:/", "")
    mv_run_info = {
        "run_id": mv_run.info.run_id,
        "best_ckpt_path": f"{artifacts_uri}/checkpoints/latest_checkpoint.pth",
    }

    return mv_run_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--alias", type=str, default=None)

    args = parser.parse_args()

    run_info = get_model_from_registry(args.model_name, args.alias)

    with mlflow.start_run(run_name="cifar_test", parent_run_id=run_info["run_id"]):

        # load config file (default config)
        cfg = get_cfg()
        cfg.data.root = '../data/'
        cfg.data.save_dir = '../logs'
        cfg.test.metrics = [AccumulatedAccuracy()]

        # initialize data module and model
        model = OsnetClassifier(cfg)
        dm = CIFARDataModule(cfg)
        dm.prepare_data()
        dm.setup(stage="test")

        callbacks = build_callbacks(cfg, logger)
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=callbacks,
            default_root_dir=cfg.data.save_dir,
        )

        result = trainer.test(
            model=model,
            ckpt_path=f"{mlflow_artifacts_path}/{run_info['best_ckpt_path']}",
            datamodule=dm
        )

        mlflow.log_metrics(result[0])
