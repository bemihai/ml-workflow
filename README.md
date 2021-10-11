## Experimental ML Workflow with `pytorch` and `mlflow`

This repo aims to create an experimental ML workflow on top of `pytorch` and `mlflow`. The main components are:

* `pytorch` - for defining ML models,
* `pytorch-lightning` - for model training/validation,
* `mlflow` - for training/validation tracking and organization, artifacts, model serving, etc.

We use `mlflow.pytorch.autolog()` which enables automatic logging of metrics, parameters, and models from a 
`pytorch-lightning` model. 

Training/validation runs are recorded as `runs` in `mlflow`. Multiple runs (e.g. multiple trainings with different 
parameters) are grouped in an `experiment`. Experiments are identified by an `int` ID and a name. `mlflow` already 
provides a default experiment with `EXPERIMENT_ID=0` and `EXPERIMENT_NAME=Default`. To create a new experiment:

```bash
 mlflow experiments create -n <EXPERIMENT_NAME>
```

To link `runs` to a specific experiments make sure you pass the corresponding experiment id:

```python
mlflow.start_run(experiment_id=EXPERIMENT_ID)
```

Alternatively, experiment name and id can be defined as environment variables: 
`MLFLOW_EXPERIMENT_NAME` and `MLFLOW_EXPERIMENT_ID`.

`mlflow` provides a web app to visualize experiments and runs. To run it locally:

```bash
mlflow ui -p <PORT>
```

To send data to this local instance of `mlflow` UI, use:

```python
mlflow.set_tracking_uri("http://localhost:<PORT>/")
```
