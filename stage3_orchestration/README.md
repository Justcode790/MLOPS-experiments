# Stage 3 – Orchestration (Prefect + Apache Airflow)

**Concept:** an orchestrator runs the pipeline steps as a **DAG of tasks** – with ordering, retries, logging, scheduling and a UI. If any task fails (e.g. the quality gate) the downstream tasks do not run.

```
ingest_and_validate → split_data → train_model → evaluate_model → quality_gate → log_to_mlflow → publish_model
```

| File | Purpose |
|---|---|
| `tasks.py` | The 7 plain-Python task functions (shared by both orchestrators; they pass data via `./work/` files) |
| `flow.py` | **Prefect** flow (retries on ingest, run logging, optional daily cron schedule) |
| `airflow_dag.py` | The same pipeline as an **Airflow** DAG (`@daily`) |
| `pipeline.py` | Shared preprocessing/model builder (same as stage 1) |

## A) Prefect (easiest – works on Windows/Mac/Linux)
```
pip install -r requirements.txt
python flow.py                                   # one run, temporary local server
```
For the dashboard: in terminal 1 `prefect server start` and in terminal 2
`set PREFECT_API_URL=http://127.0.0.1:4200/api` (PowerShell: `$env:PREFECT_API_URL="http://127.0.0.1:4200/api"`), then `python flow.py` and open **http://127.0.0.1:4200**.
Scheduled mode: `python flow.py --serve` (runs daily at 09:00).

**Failure demo (quality gate stops the flow):**
`MIN_ACCURACY=0.95 python flow.py`  (PowerShell: `$env:MIN_ACCURACY="0.95"; python flow.py`)
→ task `5-quality-gate` fails, tasks 6 and 7 never run. Unset it afterwards.

## B) Apache Airflow (Linux / macOS / WSL2 / Docker – Airflow does not run natively on Windows)
```
python -m venv af && source af/bin/activate
pip install apache-airflow mlflow pandas scikit-learn joblib
export AIRFLOW_HOME=~/airflow
mkdir -p $AIRFLOW_HOME/dags
cp airflow_dag.py tasks.py pipeline.py $AIRFLOW_HOME/dags/
cp -r ../data $AIRFLOW_HOME/dags/data
airflow standalone              # prints admin password; UI at http://localhost:8080
```
In the UI: un-pause **titanic_training_pipeline** → ▶ Trigger → open the *Graph* view and task logs.
Quick CLI check without the UI: `airflow dags test titanic_training_pipeline 2026-10-12`.

Outputs: `output/titanic_model.pkl`, `output/metrics.json`, and MLflow records in `mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
