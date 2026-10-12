# Stage 2 – MLflow (experiment tracking + model registry)

**Concept:** MLflow records *every* training run (parameters, metrics, plots, the model itself) so runs can be compared and the best model can be **registered** and loaded by name/alias – no more "which pkl was the good one?".

| File | Purpose |
|---|---|
| `train_mlflow.py` | Trains 5 candidate models (2 LogReg, 2 RandomForest, GradientBoosting); logs params, metrics, confusion-matrix plot, model + signature; registers the best (by F1) as `titanic-survival-model` with alias **@champion** |
| `compare_runs.py` | Prints all runs as a table, best first |
| `load_registered_model.py` | Loads `models:/titanic-survival-model@champion` and predicts (what a deployment would do) |

## Run
```
pip install -r requirements.txt
python train_mlflow.py
python compare_runs.py
python load_registered_model.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open **http://127.0.0.1:5001**.

**Show the TA in the UI:** Experiments → `titanic-survival` → tick all 5 runs → *Compare* (parallel-coordinates / scatter of F1 vs accuracy) · open one run → Parameters, Metrics, Artifacts (`model/`, `plots/`) · *Models* tab → `titanic-survival-model` → version 1 with alias `champion`.

Tracking data is stored in `mlflow.db` + `mlartifacts/` inside this folder (created on first run, git-ignored).
