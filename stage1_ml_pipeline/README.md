# Stage 1 – ML Pipeline (local)

**Concept:** an ML pipeline chains *ingest → preprocess → train → evaluate → save* into reproducible steps. The scikit-learn `Pipeline` bundles preprocessing and the model into **one object**, so the same transformations run at training and at prediction time.

| File | Purpose |
|---|---|
| `explore_data.py` | EDA: shape, missing values, survival by sex/class → `eda_survival.png` |
| `pipeline.py` | Reusable building blocks: `load_data`, `build_preprocessor`, `build_pipeline` |
| `train.py` | Runs the 5 pipeline steps, prints metrics, saves `titanic_model.pkl`, `metrics.json`, `confusion_matrix.png` |
| `predict_demo.py` | Loads the saved model and predicts 3 example passengers |

## Run
```
pip install -r requirements.txt
python explore_data.py
python train.py
python predict_demo.py
```
**Expected:** accuracy ≈ 0.8156, F1 ≈ 0.7179, ROC-AUC ≈ 0.8458; the 1st-class woman → SURVIVED, 3rd-class man → NOT SURVIVED.

**Show the TA:** the 5 numbered log lines of `train.py`, `confusion_matrix.png`, and that missing `Age`/`Embarked` values are handled inside the pipeline.
