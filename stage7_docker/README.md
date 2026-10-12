# Stage 7 – Docker: containerize the validated application (manual run)

**Concept:** Docker packages app + model + Python runtime + libraries into an **image**; a **container** is a running instance. (Practical 6)

| File | Purpose |
|---|---|
| `Dockerfile` | python:3.11-slim → install requirements → copy `app.py`, model, metrics → `python app.py` |
| `.dockerignore` | keeps tests/training code/git files out of the build context |
| workflow | `../.github/workflows/stage7-docker-release.yml` – release candidate **includes** Dockerfile + .dockerignore |

## Option A – from the CI artifact (as in the practical)
1. Actions → **Stage 7 - Docker-ready release candidate** → Run → download **titanic-ml-release**, extract.
2. In that folder (PowerShell):
```
docker build -t titanic-survival-ml:1.0 .
docker images
docker run -d --name titanic-survival-ml-container -p 5000:5000 titanic-survival-ml:1.0
docker ps
```
## Option B – locally from this folder
```
pip install -r requirements.txt
python train_model.py               # creates titanic_model.pkl + metrics.json that the Dockerfile copies
docker build -t titanic-survival-ml:1.0 .
docker run -d --name titanic-survival-ml-container -p 5000:5000 titanic-survival-ml:1.0
```
## Verify
Browser: http://localhost:5000/ → `{"service":"titanic-survival-prediction","status":"ok"}`

```
$body = @{ pclass=1; sex="female"; age=28; sibsp=0; parch=0; fare=100; embarked="C" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body   # SURVIVED
$body = @{ pclass=3; sex="male"; age=30; sibsp=0; parch=0; fare=7.5; embarked="S" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body   # NOT SURVIVED
docker logs titanic-survival-ml-container
```
## Port-mapping demo, stop, rerun
```
docker stop titanic-survival-ml-container ; docker rm titanic-survival-ml-container
docker run -d --name titanic-no-port titanic-survival-ml:1.0      # no -p  → localhost:5000 does NOT open
docker stop titanic-no-port ; docker rm titanic-no-port
docker run -d --name titanic-survival-ml-container -p 5000:5000 titanic-survival-ml:1.0
```
Cleanup: `docker stop titanic-survival-ml-container ; docker rm titanic-survival-ml-container`
(Port 5000 busy? use `-p 5001:5000` and browse localhost:5001.)
