# Titanic Survival – End-to-End MLOps Project (Training → Deployment)

One repository, **nine self-contained stages**. Each stage lives in its own folder with its own code, `requirements.txt`, README and (for stages 4–9) its own GitHub Actions workflow – so each stage can be run and shown to the TA separately.

**Dataset:** `data/Titanic-Dataset.csv` (891 passengers). **Task:** predict `Survived` (0/1) from Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

```
Stage 1  ML pipeline          ingest → preprocess → train → evaluate → save          (local)
Stage 2  MLflow               track 5 runs, compare, register @champion model        (local, MLflow UI)
Stage 3  Orchestration        same pipeline as a task DAG: Prefect + Airflow         (local)
Stage 4  CI + quality gate    auto train / gate / unit tests                         (GitHub Actions)
Stage 5  Artifacts            preserve model + metrics after the run                 (GitHub Actions)
Stage 6  Continuous Delivery  Flask API + API tests + release candidate              (GitHub Actions)
Stage 7  Docker               containerize, build and run by hand                    (Docker + Actions)
Stage 8  CI/CD with Docker    Actions builds image, runs container, tests API        (GitHub Actions)
Stage 9  Deployment           push validated image to GHCR, run with docker compose  (GitHub Actions + Docker)
```

| Stage | Folder | Workflow (in `.github/workflows/`) | Practical |
|---|---|---|---|
| 1 | `stage1_ml_pipeline/` | – | ML pipelines |
| 2 | `stage2_mlflow/` | – | MLflow |
| 3 | `stage3_orchestration/` | – | Orchestration tools |
| 4 | `stage4_ci_quality_gate/` | `stage4-ci-quality-gate.yml` | Practical 2–3 |
| 5 | `stage5_artifacts/` | `stage5-preserve-artifacts.yml` | Practical 4 |
| 6 | `stage6_continuous_delivery/` | `stage6-continuous-delivery.yml` | Practical 5 |
| 7 | `stage7_docker/` | `stage7-docker-release.yml` | Practical 6 |
| 8 | `stage8_docker_cicd/` | `stage8-docker-cicd.yml` | Practical 7 |
| 9 | `stage9_deployment/` | `stage9-deploy.yml` | Deployment (next step after Practical 7) |

GitHub only runs workflows stored in the repo-root `.github/workflows/`, so every workflow lives there and uses `working-directory` + `paths:` filters to work on its own stage folder. Every workflow also has a **Run workflow** button (`workflow_dispatch`).

**Start with `GUIDE.md`** – it contains the exact commands, the order, what to show the TA, and the screenshot list.
