"""Stage 3b - the same pipeline as an Apache Airflow DAG.

Copy this folder's files (airflow_dag.py, tasks.py, pipeline.py) and ../data into your Airflow dags/ folder
(see README.md), then trigger 'titanic_training_pipeline' in the Airflow UI.
"""
from datetime import datetime, timedelta

try:                                   # Airflow 3.x
    from airflow.sdk import DAG
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:                    # Airflow 2.x
    from airflow import DAG
    from airflow.operators.python import PythonOperator

import tasks

with DAG(
    dag_id="titanic_training_pipeline",
    description="Titanic survival: ingest -> split -> train -> evaluate -> gate -> MLflow -> publish",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    tags=["mlops", "titanic"],
) as dag:
    t1 = PythonOperator(task_id="ingest_and_validate", python_callable=tasks.ingest_and_validate)
    t2 = PythonOperator(task_id="split_data", python_callable=tasks.split_data)
    t3 = PythonOperator(task_id="train_model", python_callable=tasks.train_model)
    t4 = PythonOperator(task_id="evaluate_model", python_callable=tasks.evaluate_model)
    t5 = PythonOperator(task_id="quality_gate", python_callable=tasks.quality_gate)
    t6 = PythonOperator(task_id="log_to_mlflow", python_callable=tasks.log_to_mlflow)
    t7 = PythonOperator(task_id="publish_model", python_callable=tasks.publish_model)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
