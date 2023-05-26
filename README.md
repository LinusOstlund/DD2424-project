# Project for DD2424 Deep Learning in Data Science
Authors: Linus Östlund and Marco Amoretti

## Project description

* Something on the dataset (cite their work)
* Something on the architecture
* Something on Molly

## How to use with MLFlow

### Create Virtual Environment

1. Install MLFlow:

```python
pip install mlflow
```

2. Open the MLFlow UI in a new terminal. By default the following command opens a GUI at `http://localhost:5000`:

```bash
mlflow ui
```

3. Run the project by copy-pasting the command below in a fresh terminal. You are going to install a virtual environment, which is cached for future testing.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
 mlflow run git@github.com:LinusOstlund/DD2424-project.git 
```
Once the run is complete, you may open the MLFlow GUI and trace the performance. The ABC-notation is saved under the folder `inference`.

> **NOTE:** as per [this issue](https://github.com/mlflow/mlflow/issues/608#issuecomment-454316004), the MLFlow Tracking URI has to be set as an environment variable. Odd, and workaround-ish.

4. **OPTIONAL**: Test different parameter settings. You may see the full list in [MLproject](/MLproject). For instance, these settings turned out quite good.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
 mlflow run git@github.com:LinusOstlund/DD2424-project.git \
 -P epochs=200 \
 -P hidden_size=64 \
 -P embedding_size=64 \
 -P batch_size=32 \
 -P learning_rate=0.001 \
 -P mode="topk"
```

### TODO
* tidy up...
* fix so the midi is saved, too
* fix the "generate data" code, so we can generate it from scratch
* load a checkpoint to avoid training all the time
* fix the transformer model

