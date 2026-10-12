"""Stage 3a - Orchestration with Prefect.

Run once :   python flow.py
Run daily :  python flow.py --serve     (keeps running; cron schedule 09:00 every day)
Dashboard :  prefect server start       (then open http://127.0.0.1:4200 before running the flow)
"""
import sys

from prefect import flow, get_run_logger, task

import tasks

ingest = task(tasks.ingest_and_validate, name="1-ingest-and-validate", retries=2, retry_delay_seconds=2)
split = task(tasks.split_data, name="2-split-data")
train = task(tasks.train_model, name="3-train-model")
evaluate = task(tasks.evaluate_model, name="4-evaluate-model")
gate = task(tasks.quality_gate, name="5-quality-gate")
log_mlflow = task(tasks.log_to_mlflow, name="6-log-to-mlflow")
publish = task(tasks.publish_model, name="7-publish-model")


@flow(name="titanic-training-pipeline", log_prints=True)
def titanic_training_pipeline():
    logger = get_run_logger()
    logger.info("Pipeline started")
    ingest()
    split()
    train()
    metrics = evaluate()
    gate()                     # raises -> flow FAILS and the next tasks never run
    log_mlflow()
    out = publish()
    logger.info("Pipeline finished, model published to %s", out)
    return metrics


if __name__ == "__main__":
    if "--serve" in sys.argv:
        titanic_training_pipeline.serve(name="daily-titanic-training", cron="0 9 * * *")
    else:
        print(titanic_training_pipeline())
