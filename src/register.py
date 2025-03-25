import argparse

import mlflow
from mlflow.entities import RunStatus


mlflow.tracking.set_tracking_uri("http://localhost:5000")


def register_model(run_id: str, model_name: str, tags: dict[str, str] | None = None, alias: str | None = None):
    """Register a new model version to the MLFlow's Model Registry."""
    # get the training run info from the db
    client = mlflow.tracking.MlflowClient()
    run_info = client.get_run(run_id)

    # check if the training run is finished, otherwise raise error
    run_status = RunStatus.from_string(run_info.info.status)
    if run_status != RunStatus.FINISHED:
        raise ValueError(f"Training run is not finished (status: {run_info.info.status})")

    # register model version
    # tags = {**tags, **run_info.data.tags} if tags else run_info.data.tags
    tags = tags or {}
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_info.info.run_id}/model",
        name=model_name,
        tags=tags,
    )

    # set alias if provided
    if alias:
        client.set_registered_model_alias(model_version.name, alias, model_version.version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
    args = parser.parse_args()

    register_model(args.run_id, args.model_name, args.tags, args.stage)