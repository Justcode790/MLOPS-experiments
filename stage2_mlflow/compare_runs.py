"""Print all runs of the experiment as a table, best first."""
import mlflow

from pipeline import HERE

mlflow.set_tracking_uri(f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
runs = mlflow.search_runs(experiment_names=["titanic-survival"], order_by=["metrics.f1 DESC"])
cols = ["tags.mlflow.runName", "params.algorithm", "metrics.accuracy", "metrics.f1", "metrics.roc_auc"]
print(runs[cols].rename(columns=lambda c: c.split(".")[-1]).to_string(index=False))
