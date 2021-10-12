## Experimental ML Workflow with `pytorch` and `mlflow`

### ML Workflow 

This repo aims to create an experimental ML workflow on top of `pytorch` and `mlflow`. The main components are:

* `pytorch` - for defining ML models,
* `pytorch-lightning` - for model training/validation,
* `mlflow` - for training/validation tracking and organization, artifacts, model serving, etc.

We use `mlflow.pytorch.autolog()` which enables automatic logging of metrics, parameters, and models from a 
`pytorch-lightning` model. 

Training/testing runs are recorded as `runs` in `mlflow`. Multiple runs (e.g. multiple trainings with different 
parameters) are grouped under an `experiment`. Experiments are identified by an `int` ID and a name. `mlflow` already 
provides a default experiment with `EXPERIMENT_ID=0` and `EXPERIMENT_NAME=Default`. 

To create a new experiment from the `mlflow` CLI:

```bash
 mlflow experiments create -n <EXPERIMENT_NAME>
```

To link `runs` to a specific experiment use the Python API and make sure you pass the corresponding experiment id:

```python
mlflow.start_run(experiment_id=EXPERIMENT_ID)
```

Alternatively, experiment name and id can be defined as environment variables: 
`MLFLOW_EXPERIMENT_NAME` and `MLFLOW_EXPERIMENT_ID`.

### Running the code

To run the code from this example, go to the `ml-workflow` folder and run:

```bash
python src/main.py --run-name <RUN NAME> --experiment-id <ID> 
```

### View results in `mlflow` UI

`mlflow` provides a web app to visualize experiments and runs. To run it locally:

```bash
mlflow ui
```

To skip the ssl certificate verification, install a new certificate using 
`./Install Certificates.command` which is in the refreshed Python 3.8 directory:
```bash
cd "/Applications/Python 3.8/"
sudo "./Install Certificates.command"
```
