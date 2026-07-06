# Dashboard Final ALDIMI Predict — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Hito 4 "Dashboard Final" — a local Streamlit app with 4 pages (Alertas de Stock, Riesgo de Pacientes, Interpretabilidad, Métricas del Modelo) backed by the winning models from Hito 3, persisted via a standalone training script instead of being retrained inline in a notebook.

**Architecture:** `codigo/train_models.py` reproduces the exact Hito 3 pipeline (SMOTE + RandomForest/XGBoost for risk classification, chronological split + RandomForest/XGBoost for stock regression) and persists the winning estimator per task with `joblib`, plus a `modelos/metricas.json` with evaluation metrics. `codigo/app.py` + `codigo/pages/*.py` is a Streamlit multipage app that only loads artifacts and renders them — no training happens at dashboard runtime. Business logic (alert thresholds, risk-table construction) lives in small pure-Python modules under `codigo/utils/` so it's unit-testable without spinning up Streamlit.

**Tech Stack:** Python 3.9+, scikit-learn 1.7.0, XGBoost 3.1.2, imbalanced-learn 0.14.1 (SMOTE), joblib 1.5.1, Streamlit 1.45.1, Plotly 6.1.2, pytest 9.0.1 (including `streamlit.testing.v1.AppTest` for headless page tests).

## Global Constraints

- **Commit locally at the end of each task, but never push and never add a `Co-Authored-By` trailer.** This plan executes inside an isolated worktree (branch `worktree-dashboard-final-hito4`) specifically so per-task commits are safe and diffable for review. The user will decide separately whether/when to push or merge — never run `git push` as part of executing this plan, and never include a `Co-Authored-By: Claude ...` line in these commit messages (explicit user instruction, so their participation doesn't show if the branch is ever pushed).
- **Reproduce the Hito 3 stock-regressor pipeline exactly, including its known gap:** train on `FEATURE_COLS_INV` (no `Stock_Lag1`/`Stock_Lag7`/`Stock_RollingMean7`), matching what was already evaluated and presented in Hito 3 — user explicitly chose not to fix this now (see `docs/superpowers/specs/2026-07-06-dashboard-final-design.md`).
- **Datasets stay at the repo root** (`cancer-risk-factors.csv`, `inventory_data.csv`) — do not move or copy them into `datos/` as part of this plan. All scripts reference them via a `REPO_ROOT`-relative path. Physical packaging into `/codigo /datos /docs` for the submission zip is a separate, later step out of scope here.
- **Dashboard runs only locally** (`streamlit run codigo/app.py`) — no hosting/deployment config.
- **No `__init__.py` files** under `codigo/utils/` or `codigo/tests/` — Python 3 implicit namespace packages plus `codigo/conftest.py` (which inserts `codigo/` onto `sys.path`) are enough to make `from utils.x import y` resolve in tests, scripts, and Streamlit pages alike.
- Package versions in `requirements.txt` are pinned to what's already installed and verified working in this environment (`pip show`): pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.0, xgboost 3.1.2, imbalanced-learn 0.14.1, joblib 1.5.1, streamlit 1.45.1, plotly 6.1.2, pytest 9.0.1.

---

### Task 1: Scaffolding — folders, requirements.txt, conftest.py, .gitignore

**Files:**
- Create: `requirements.txt`
- Create: `codigo/conftest.py`
- Create: `.gitignore`
- Create: `modelos/.gitkeep`
- Create: `codigo/utils/` and `codigo/pages/` and `codigo/tests/` (empty directories, populated in later tasks)

**Interfaces:**
- Produces: `codigo/` on `sys.path` for every test file (via `conftest.py`), so later tasks can write `from utils.preprocessing import ...` etc. without repeating path hacks in tests.

- [ ] **Step 1: Create `requirements.txt`**

```
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.0
xgboost==3.1.2
imbalanced-learn==0.14.1
joblib==1.5.1
streamlit==1.45.1
plotly==6.1.2
pytest==9.0.1
```

- [ ] **Step 2: Create `codigo/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/

# Modelos: regenerados por codigo/train_models.py, no se versionan
modelos/*
!modelos/.gitkeep
```

- [ ] **Step 4: Create the folders and a placeholder file**

```bash
mkdir -p codigo/utils codigo/pages codigo/tests modelos
touch modelos/.gitkeep
```

- [ ] **Step 5: Verify scaffolding**

Run: `python -c "import pandas, numpy, sklearn, xgboost, imblearn, joblib, streamlit, plotly, pytest; print('OK')"`
Expected: `OK` (confirms every pinned package is importable in this environment before writing code against it)

---

### Task 2: `codigo/utils/preprocessing.py` — shared feature engineering

**Files:**
- Create: `codigo/utils/preprocessing.py`
- Test: `codigo/tests/test_preprocessing.py`

**Interfaces:**
- Produces: `engineer_health_features(df_cancer, cancer_type_encoder=None) -> (pd.DataFrame, LabelEncoder)`, `engineer_inventory_features(df_inv, item_name_encoder=None, item_type_encoder=None) -> (pd.DataFrame, LabelEncoder, LabelEncoder)`, `FEATURE_COLS_CANCER: list[str]`, `FEATURE_COLS_INV: list[str]`. Consumed by `train_models.py` (Task 3) and `utils/stock_alerts.py` / `utils/patient_risk.py` (Tasks 5-6).

- [ ] **Step 1: Write the failing tests**

```python
# codigo/tests/test_preprocessing.py
from pathlib import Path

import pandas as pd

from utils.preprocessing import (
    engineer_health_features,
    engineer_inventory_features,
    FEATURE_COLS_CANCER,
    FEATURE_COLS_INV,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_engineer_health_features_computes_derived_columns():
    df = pd.DataFrame({
        'Cancer_Type': ['Breast', 'Prostate', 'Breast'],
        'Smoking': [1, 0, 1],
        'Alcohol_Use': [0, 1, 1],
        'Air_Pollution': [1, 1, 0],
        'Diet_Salted_Processed': [2, 1, 3],
        'Diet_Red_Meat': [1, 2, 1],
        'Fruit_Veg_Intake': [3, 2, 1],
        'BRCA_Mutation': [0, 1, 0],
        'Family_History': [1, 1, 0],
        'H_Pylori_Infection': [0, 0, 1],
    })

    result, encoder = engineer_health_features(df)

    assert list(result['Risk_Lifestyle']) == [2, 2, 2]
    assert list(result['Diet_Score']) == [0, 1, 3]
    assert list(result['Genetic_Risk']) == [1, 2, 1]
    assert result['Cancer_Type_enc'].tolist() == encoder.transform(df['Cancer_Type']).tolist()


def test_engineer_health_features_reuses_fitted_encoder():
    df_train = pd.DataFrame({'Cancer_Type': ['Breast', 'Prostate']})
    for col in ['Smoking', 'Alcohol_Use', 'Air_Pollution', 'Diet_Salted_Processed',
                'Diet_Red_Meat', 'Fruit_Veg_Intake', 'BRCA_Mutation', 'Family_History',
                'H_Pylori_Infection']:
        df_train[col] = [0, 0]

    _, encoder = engineer_health_features(df_train)
    result_new, encoder_reused = engineer_health_features(df_train.copy(), cancer_type_encoder=encoder)

    assert encoder_reused is encoder
    assert result_new['Cancer_Type_enc'].tolist() == encoder.transform(df_train['Cancer_Type']).tolist()


def test_engineer_inventory_features_computes_derived_columns():
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-08'],
        'Item_Name': ['Guantes', 'Guantes'],
        'Item_Type': ['Insumo Medico', 'Insumo Medico'],
        'Current_Stock': [500, 400],
        'Min_Required': [300, 300],
        'Avg_Usage_Per_Day': [20, 20],
    })

    result, item_name_enc, item_type_enc = engineer_inventory_features(df)

    assert list(result['Month']) == [1, 1]
    assert list(result['Projected_Stock_7d']) == [360, 260]
    assert list(result['Projected_Stock_14d']) == [220, 120]
    assert list(result['Stock_Ratio']) == [round(500 / 300, 3), round(400 / 300, 3)]
    assert 'Stock_Lag1' not in result.columns  # decision: no lag features (matches Hito 3)


def test_feature_cols_present_after_engineering_on_real_data():
    df_cancer = pd.read_csv(REPO_ROOT / 'cancer-risk-factors.csv')
    result, _ = engineer_health_features(df_cancer)
    assert all(col in result.columns for col in FEATURE_COLS_CANCER)

    df_inv = pd.read_csv(REPO_ROOT / 'inventory_data.csv')
    result_inv, _, _ = engineer_inventory_features(df_inv)
    assert all(col in result_inv.columns for col in FEATURE_COLS_INV)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codigo/tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils'` (module doesn't exist yet)

- [ ] **Step 3: Write `codigo/utils/preprocessing.py`**

```python
"""Ingenieria de caracteristicas compartida (Hito 2/3) para ambos frentes.

Funciones puras: reciben y devuelven DataFrames, no leen archivos.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS_CANCER = [
    'Age', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity', 'Family_History',
    'Diet_Red_Meat', 'Diet_Salted_Processed', 'Fruit_Veg_Intake',
    'Physical_Activity', 'Air_Pollution', 'Occupational_Hazards',
    'BRCA_Mutation', 'H_Pylori_Infection', 'Calcium_Intake',
    'BMI', 'Physical_Activity_Level', 'Cancer_Type_enc',
    'Risk_Lifestyle', 'Diet_Score', 'Genetic_Risk',
    # Overall_Risk_Score y Metastasis_Status excluidas: data leakage (Hito 2)
]

FEATURE_COLS_INV = [
    'Current_Stock', 'Min_Required', 'Max_Capacity', 'Unit_Cost',
    'Avg_Usage_Per_Day', 'Restock_Lead_Time',
    'Item_Name_enc', 'Item_Type_enc',
    'Month', 'Quarter', 'Day_of_Week',
    'Stock_Ratio', 'Days_Until_Stockout',
]


def engineer_health_features(df_cancer: pd.DataFrame, cancer_type_encoder: LabelEncoder = None):
    """Reproduce la ingenieria de caracteristicas de Hito 2/3 para el frente de salud.

    Si cancer_type_encoder es None se ajusta uno nuevo (modo entrenamiento);
    si se pasa uno ya ajustado se reutiliza (modo inferencia), para garantizar
    la misma codificacion Cancer_Type -> entero entre train y predict.
    """
    df = df_cancer.copy()
    df['Risk_Lifestyle'] = df['Smoking'] + df['Alcohol_Use'] + df['Air_Pollution']
    df['Diet_Score'] = df['Diet_Salted_Processed'] + df['Diet_Red_Meat'] - df['Fruit_Veg_Intake']
    df['Genetic_Risk'] = df['BRCA_Mutation'] + df['Family_History'] + df['H_Pylori_Infection']

    if cancer_type_encoder is None:
        cancer_type_encoder = LabelEncoder()
        df['Cancer_Type_enc'] = cancer_type_encoder.fit_transform(df['Cancer_Type'])
    else:
        df['Cancer_Type_enc'] = cancer_type_encoder.transform(df['Cancer_Type'])

    return df, cancer_type_encoder


def engineer_inventory_features(df_inv: pd.DataFrame, item_name_encoder: LabelEncoder = None,
                                 item_type_encoder: LabelEncoder = None):
    """Reproduce la ingenieria de caracteristicas de Hito 2/3 para el frente logistico.

    No incluye features de lag temporal (Stock_Lag1/7, RollingMean7): en Hito 3
    se calcularon pero el entrenamiento del regresor final uso FEATURE_COLS_INV
    (sin lag), no FEATURE_COLS_INV_TS. Se reproduce tal cual para no alterar las
    metricas ya sustentadas (decision confirmada para Hito 4).
    """
    df = df_inv.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    df['Projected_Stock_7d'] = (df['Current_Stock'] - df['Avg_Usage_Per_Day'] * 7).clip(lower=0)
    df['Projected_Stock_14d'] = (df['Current_Stock'] - df['Avg_Usage_Per_Day'] * 14).clip(lower=0)
    df['Days_Until_Stockout'] = (df['Current_Stock'] / df['Avg_Usage_Per_Day']).round(2)
    df['Stock_Ratio'] = (df['Current_Stock'] / df['Min_Required']).round(3)

    if item_name_encoder is None:
        item_name_encoder = LabelEncoder()
        df['Item_Name_enc'] = item_name_encoder.fit_transform(df['Item_Name'])
    else:
        df['Item_Name_enc'] = item_name_encoder.transform(df['Item_Name'])

    if item_type_encoder is None:
        item_type_encoder = LabelEncoder()
        df['Item_Type_enc'] = item_type_encoder.fit_transform(df['Item_Type'])
    else:
        df['Item_Type_enc'] = item_type_encoder.transform(df['Item_Type'])

    return df, item_name_encoder, item_type_encoder
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codigo/tests/test_preprocessing.py -v`
Expected: PASS (5 tests)

---

### Task 3: `codigo/train_models.py` — train and score both fronts

**Files:**
- Create: `codigo/train_models.py`
- Test: `codigo/tests/test_train_models.py`

**Interfaces:**
- Consumes: `utils.preprocessing.engineer_health_features`, `engineer_inventory_features`, `FEATURE_COLS_CANCER`, `FEATURE_COLS_INV` (Task 2).
- Produces: `train_risk_classifier(df_cancer_raw) -> (model, cancer_type_encoder, target_encoder, metrics: dict)` and `train_stock_regressor(df_inventory_raw, horizon: int) -> (model, item_name_encoder, item_type_encoder, metrics: dict)`. Consumed by `main()` in this same file (Task 4 runs it) — no other task calls these functions directly.

- [ ] **Step 1: Write the failing tests**

```python
# codigo/tests/test_train_models.py
from pathlib import Path

import pandas as pd
import pytest

from train_models import train_risk_classifier, train_stock_regressor

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_train_risk_classifier_returns_valid_model_and_metrics():
    df_cancer = pd.read_csv(REPO_ROOT / 'cancer-risk-factors.csv')

    model, cancer_type_encoder, target_encoder, metrics = train_risk_classifier(df_cancer)

    assert hasattr(model, 'predict')
    assert metrics['modelo_ganador'] in ('Random Forest', 'XGBoost')
    assert 0.0 <= metrics['f1_macro'] <= 1.0
    assert 0.0 <= metrics['f1_high'] <= 1.0
    assert set(target_encoder.classes_) == {'Low', 'Medium', 'High'}
    assert len(metrics['matriz_confusion']) == 3


def test_train_stock_regressor_returns_valid_model_and_metrics_for_each_horizon():
    df_inv = pd.read_csv(REPO_ROOT / 'inventory_data.csv')

    for horizon in (7, 14):
        model, item_name_enc, item_type_enc, metrics = train_stock_regressor(df_inv, horizon)

        assert hasattr(model, 'predict')
        assert metrics['modelo_ganador'] in ('Random Forest', 'XGBoost')
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0


def test_train_stock_regressor_rejects_invalid_horizon():
    df_inv = pd.read_csv(REPO_ROOT / 'inventory_data.csv')

    with pytest.raises(AssertionError):
        train_stock_regressor(df_inv, horizon=30)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codigo/tests/test_train_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_models'`

- [ ] **Step 3: Write `codigo/train_models.py`**

```python
"""Entrena los modelos finales de ALDIMI Predict (Hito 3) y los persiste en /modelos.

Reproduce el pipeline de Hito 3.ipynb (mismos hiperparametros de busqueda,
mismos splits, mismas features) para no alterar las metricas ya sustentadas.

Ejecutar una sola vez (o cuando cambien los datos de entrenamiento):
    python codigo/train_models.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from utils.preprocessing import (
    FEATURE_COLS_CANCER,
    FEATURE_COLS_INV,
    engineer_health_features,
    engineer_inventory_features,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELOS_DIR = REPO_ROOT / "modelos"


def train_risk_classifier(df_cancer_raw: pd.DataFrame):
    """SMOTE + mejor de (RandomForest GridSearchCV, XGBoost RandomizedSearchCV).

    Retorna (modelo_ganador, cancer_type_encoder, target_encoder, metricas_dict).
    """
    df, cancer_type_encoder = engineer_health_features(df_cancer_raw)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['Risk_Level'])
    X = df[FEATURE_COLS_CANCER]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced'],
        },
        cv=cv_strat, scoring='f1_macro', n_jobs=-1,
    )
    gs_rf.fit(X_train_sm, y_train_sm)
    best_rf = gs_rf.best_estimator_
    rf_f1 = f1_score(y_test, best_rf.predict(X_test), average='macro')

    rs_xgb = RandomizedSearchCV(
        XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss',
                      random_state=42, verbosity=0),
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'min_child_weight': [1, 3, 5],
        },
        n_iter=20, cv=cv_strat, scoring='f1_macro', n_jobs=-1, random_state=42,
    )
    rs_xgb.fit(X_train_sm, y_train_sm)
    best_xgb = rs_xgb.best_estimator_
    xgb_f1 = f1_score(y_test, best_xgb.predict(X_test), average='macro')

    if xgb_f1 >= rf_f1:
        winner, winner_name, winner_f1 = best_xgb, 'XGBoost', xgb_f1
    else:
        winner, winner_name, winner_f1 = best_rf, 'Random Forest', rf_f1

    y_pred = winner.predict(X_test)
    idx_high = list(target_encoder.classes_).index('High')
    metrics = {
        'modelo_ganador': winner_name,
        'f1_macro': float(winner_f1),
        'f1_high': float(f1_score(y_test, y_pred, average=None)[idx_high]),
        'matriz_confusion': confusion_matrix(y_test, y_pred).tolist(),
        'clases': target_encoder.classes_.tolist(),
        'feature_importances': (
            dict(zip(FEATURE_COLS_CANCER, winner.feature_importances_.tolist()))
            if hasattr(winner, 'feature_importances_') else None
        ),
    }
    return winner, cancer_type_encoder, target_encoder, metrics


def train_stock_regressor(df_inventory_raw: pd.DataFrame, horizon: int):
    """Split temporal cronologico + mejor de (RF GridSearchCV, XGBoost RandomizedSearchCV).

    horizon: 7 o 14. Sin features de lag (ver Global Constraints del plan).
    Retorna (modelo_ganador, item_name_encoder, item_type_encoder, metricas_dict).
    """
    assert horizon in (7, 14), "horizon debe ser 7 o 14"
    target_col = f'Projected_Stock_{horizon}d'

    df, item_name_encoder, item_type_encoder = engineer_inventory_features(df_inventory_raw)
    df_sorted = df.sort_values('Date').reset_index(drop=True)

    cut = int(len(df_sorted) * 0.8)
    X_train = df_sorted[FEATURE_COLS_INV].iloc[:cut]
    X_test = df_sorted[FEATURE_COLS_INV].iloc[cut:]
    y_train = df_sorted[target_col].iloc[:cut]
    y_test = df_sorted[target_col].iloc[cut:]

    gs_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
    )
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    rf_mae = mean_absolute_error(y_test, best_rf.predict(X_test))

    rs_xgb = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
        },
        n_iter=15, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42,
    )
    rs_xgb.fit(X_train, y_train)
    best_xgb = rs_xgb.best_estimator_
    xgb_mae = mean_absolute_error(y_test, best_xgb.predict(X_test))

    if xgb_mae <= rf_mae:
        winner, winner_name, winner_mae = best_xgb, 'XGBoost', xgb_mae
    else:
        winner, winner_name, winner_mae = best_rf, 'Random Forest', rf_mae

    y_pred = winner.predict(X_test)
    metrics = {
        'modelo_ganador': winner_name,
        'mae': float(winner_mae),
        'r2': float(r2_score(y_test, y_pred)),
        'feature_importances': (
            dict(zip(FEATURE_COLS_INV, winner.feature_importances_.tolist()))
            if hasattr(winner, 'feature_importances_') else None
        ),
    }
    return winner, item_name_encoder, item_type_encoder, metrics


def main():
    MODELOS_DIR.mkdir(exist_ok=True)

    df_cancer = pd.read_csv(REPO_ROOT / 'cancer-risk-factors.csv')
    df_inv = pd.read_csv(REPO_ROOT / 'inventory_data.csv')

    print("Entrenando clasificador de riesgo de pacientes...")
    model_risk, cancer_type_enc, target_enc, metrics_risk = train_risk_classifier(df_cancer)
    joblib.dump({
        'model': model_risk,
        'cancer_type_encoder': cancer_type_enc,
        'target_encoder': target_enc,
        'feature_cols': FEATURE_COLS_CANCER,
    }, MODELOS_DIR / 'riesgo_paciente.pkl')
    print(f"  Ganador: {metrics_risk['modelo_ganador']} (F1-macro={metrics_risk['f1_macro']:.4f})")

    metricas_json = {'riesgo_paciente': metrics_risk}

    for horizon in (7, 14):
        print(f"Entrenando regresor de stock a {horizon} dias...")
        model_stock, item_name_enc, item_type_enc, metrics_stock = train_stock_regressor(df_inv, horizon)
        joblib.dump({
            'model': model_stock,
            'item_name_encoder': item_name_enc,
            'item_type_encoder': item_type_enc,
            'feature_cols': FEATURE_COLS_INV,
        }, MODELOS_DIR / f'stock_{horizon}d.pkl')
        print(f"  Ganador: {metrics_stock['modelo_ganador']} "
              f"(MAE={metrics_stock['mae']:.1f}, R2={metrics_stock['r2']:.4f})")
        metricas_json[f'stock_{horizon}d'] = metrics_stock

    with open(MODELOS_DIR / 'metricas.json', 'w', encoding='utf-8') as f:
        json.dump(metricas_json, f, indent=2, ensure_ascii=False)
    print(f"\nArtefactos guardados en {MODELOS_DIR}/")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codigo/tests/test_train_models.py -v`
Expected: PASS (3 tests). This actually runs GridSearchCV/RandomizedSearchCV on the real (small) datasets, so expect ~30-90 seconds, not instant.

---

### Task 4: `codigo/utils/model_io.py` + generate real model artifacts

**Files:**
- Create: `codigo/utils/model_io.py`
- Test: `codigo/tests/test_model_io.py`

**Interfaces:**
- Consumes: nothing (reads from `MODELOS_DIR` produced by `train_models.py`, Task 3).
- Produces: `load_bundle(name: str) -> dict` and `load_metrics() -> dict`. Consumed by every dashboard page (Tasks 7-11).

- [ ] **Step 1: Write `codigo/utils/model_io.py`**

```python
"""Carga de artefactos persistidos por train_models.py.

Seguridad: joblib.load usa pickle internamente (ejecucion de codigo arbitrario
si el archivo viniera de una fuente no confiable). Aqui es seguro: los .pkl en
modelos/ los genera exclusivamente train_models.py en la misma maquina/repo, no
se descargan ni se aceptan de terceros.
"""
import json
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOS_DIR = REPO_ROOT / "modelos"


def load_bundle(name: str) -> dict:
    """Carga un artefacto (ej. 'stock_7d', 'stock_14d', 'riesgo_paciente')."""
    return joblib.load(MODELOS_DIR / f"{name}.pkl")


def load_metrics() -> dict:
    with open(MODELOS_DIR / "metricas.json", encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 2: Generate the real artifacts by running the training script**

Run: `python codigo/train_models.py`
Expected: prints progress for the 3 models and ends with `Artefactos guardados en .../modelos/`; creates `modelos/riesgo_paciente.pkl`, `modelos/stock_7d.pkl`, `modelos/stock_14d.pkl`, `modelos/metricas.json`.

- [ ] **Step 3: Write the test for the loaders**

```python
# codigo/tests/test_model_io.py
import pytest

from utils.model_io import MODELOS_DIR, load_bundle, load_metrics

pytestmark = pytest.mark.skipif(
    not (MODELOS_DIR / 'metricas.json').exists(),
    reason="Requiere ejecutar antes: python codigo/train_models.py",
)


def test_load_bundle_returns_dict_with_model_and_encoders():
    bundle = load_bundle('riesgo_paciente')
    assert hasattr(bundle['model'], 'predict')
    assert 'cancer_type_encoder' in bundle
    assert 'target_encoder' in bundle
    assert 'feature_cols' in bundle


def test_load_metrics_has_all_three_models():
    metrics = load_metrics()
    assert set(metrics.keys()) == {'riesgo_paciente', 'stock_7d', 'stock_14d'}
    assert 'f1_macro' in metrics['riesgo_paciente']
    assert 'mae' in metrics['stock_7d']
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codigo/tests/test_model_io.py -v`
Expected: PASS (2 tests)

---

### Task 5: `codigo/utils/stock_alerts.py` — stock alert business logic

**Files:**
- Create: `codigo/utils/stock_alerts.py`
- Test: `codigo/tests/test_stock_alerts.py`

**Interfaces:**
- Consumes: `utils.preprocessing.engineer_inventory_features`, `FEATURE_COLS_INV` (Task 2). Takes model bundles shaped like `utils.model_io.load_bundle('stock_7d')` output (Task 4) but does not import `model_io` itself — kept a pure function for testability.
- Produces: `semaforo(pred: float, min_required: float) -> str`, `build_alert_table(df_inv, bundle_7, bundle_14) -> pd.DataFrame`. Consumed by `pages/1_Alertas_Stock.py` (Task 8).

- [ ] **Step 1: Write the failing tests**

```python
# codigo/tests/test_stock_alerts.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.stock_alerts import build_alert_table, semaforo


class _FixedPredictor:
    """Estimador falso que siempre predice el mismo valor, para aislar la
    logica de semaforo/tabla de la logica real de ML (probada en Task 3)."""
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


def test_semaforo_thresholds():
    assert semaforo(pred=50, min_required=100) == "Crítico"
    assert semaforo(pred=120, min_required=100) == "Atención"
    assert semaforo(pred=140, min_required=100) == "OK"


def test_build_alert_table_uses_most_recent_row_per_item_and_flags_states():
    df_inv = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Item_Name': ['Guantes', 'Guantes'],
        'Item_Type': ['Insumo Medico', 'Insumo Medico'],
        'Current_Stock': [500, 400],
        'Min_Required': [300, 300],
        'Max_Capacity': [1000, 1000],
        'Unit_Cost': [1.0, 1.0],
        'Avg_Usage_Per_Day': [20, 20],
        'Restock_Lead_Time': [5, 5],
        'Vendor_ID': ['V001', 'V001'],
    })
    item_name_encoder = LabelEncoder().fit(df_inv['Item_Name'])
    item_type_encoder = LabelEncoder().fit(df_inv['Item_Type'])

    bundle_7 = {'model': _FixedPredictor(250), 'item_name_encoder': item_name_encoder,
                'item_type_encoder': item_type_encoder}
    bundle_14 = {'model': _FixedPredictor(350), 'item_name_encoder': item_name_encoder,
                 'item_type_encoder': item_type_encoder}

    result = build_alert_table(df_inv, bundle_7, bundle_14)

    assert len(result) == 1  # un registro por item (el mas reciente)
    row = result.iloc[0]
    assert row['Current_Stock'] == 400  # fila del 2024-01-02, no la del 01
    assert row['Estado_7d'] == "Crítico"    # 250 < 300
    assert row['Estado_14d'] == "Atención"  # 300 <= 350 < 390
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest codigo/tests/test_stock_alerts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.stock_alerts'`

- [ ] **Step 3: Write `codigo/utils/stock_alerts.py`**

```python
"""Logica de negocio para la pagina de Alertas de Stock."""
import pandas as pd

from utils.preprocessing import FEATURE_COLS_INV, engineer_inventory_features


def semaforo(pred: float, min_required: float) -> str:
    """Clasifica una prediccion de stock segun el margen sobre Min_Required."""
    if pred < min_required:
        return "Crítico"
    if pred < min_required * 1.3:
        return "Atención"
    return "OK"


def build_alert_table(df_inv: pd.DataFrame, bundle_7: dict, bundle_14: dict) -> pd.DataFrame:
    """Tabla de alertas con predicciones a 7 y 14 dias para el registro mas
    reciente de cada insumo."""
    df_feat, _, _ = engineer_inventory_features(
        df_inv,
        item_name_encoder=bundle_7['item_name_encoder'],
        item_type_encoder=bundle_7['item_type_encoder'],
    )
    latest = (df_feat.sort_values('Date')
              .groupby('Item_Name', as_index=False)
              .last())

    X = latest[FEATURE_COLS_INV]
    latest['Pred_7d'] = bundle_7['model'].predict(X)
    latest['Pred_14d'] = bundle_14['model'].predict(X)
    latest['Estado_7d'] = latest.apply(lambda r: semaforo(r['Pred_7d'], r['Min_Required']), axis=1)
    latest['Estado_14d'] = latest.apply(lambda r: semaforo(r['Pred_14d'], r['Min_Required']), axis=1)

    return latest[['Item_Name', 'Item_Type', 'Vendor_ID', 'Current_Stock',
                    'Min_Required', 'Pred_7d', 'Estado_7d', 'Pred_14d', 'Estado_14d']]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest codigo/tests/test_stock_alerts.py -v`
Expected: PASS (2 tests)

---

### Task 6: `codigo/utils/patient_risk.py` — patient risk table business logic

**Files:**
- Create: `codigo/utils/patient_risk.py`
- Test: `codigo/tests/test_patient_risk.py`

**Interfaces:**
- Consumes: `utils.preprocessing.engineer_health_features`, `FEATURE_COLS_CANCER` (Task 2).
- Produces: `build_risk_table(df_cancer, bundle) -> pd.DataFrame`. Consumed by `pages/2_Riesgo_Pacientes.py` (Task 9).

- [ ] **Step 1: Write the failing test**

```python
# codigo/tests/test_patient_risk.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.patient_risk import build_risk_table


class _FixedClassifier:
    def __init__(self, encoded_values):
        self.encoded_values = encoded_values

    def predict(self, X):
        return self.encoded_values[:len(X)]


def test_build_risk_table_orders_high_risk_first():
    df_cancer = pd.DataFrame({
        'Patient_ID': ['P1', 'P2', 'P3'],
        'Cancer_Type': ['Breast', 'Breast', 'Lung'],
        'Age': [40, 55, 60],
        'Gender': [0, 1, 0],
        'Smoking': [0, 1, 1], 'Alcohol_Use': [0, 0, 1], 'Obesity': [0, 1, 0],
        'Family_History': [0, 1, 1], 'Diet_Red_Meat': [1, 2, 3],
        'Diet_Salted_Processed': [1, 2, 1], 'Fruit_Veg_Intake': [3, 1, 1],
        'Physical_Activity': [2, 1, 1], 'Air_Pollution': [1, 1, 1],
        'Occupational_Hazards': [0, 0, 1], 'BRCA_Mutation': [0, 0, 1],
        'H_Pylori_Infection': [0, 0, 0], 'Calcium_Intake': [2, 2, 1],
        'BMI': [22.0, 27.0, 30.0], 'Physical_Activity_Level': [1, 1, 0],
    })

    cancer_type_encoder = LabelEncoder().fit(df_cancer['Cancer_Type'])
    target_encoder = LabelEncoder().fit(['High', 'Low', 'Medium'])
    # target_encoder.classes_ ordenadas alfabeticamente: High=0, Low=1, Medium=2
    bundle = {
        'model': _FixedClassifier([1, 2, 0]),  # P1->Low, P2->Medium, P3->High
        'cancer_type_encoder': cancer_type_encoder,
        'target_encoder': target_encoder,
    }

    result = build_risk_table(df_cancer, bundle)

    assert list(result['Patient_ID']) == ['P3', 'P2', 'P1']
    assert result.iloc[0]['Risk_Level_Predicho'] == 'High'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest codigo/tests/test_patient_risk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.patient_risk'`

- [ ] **Step 3: Write `codigo/utils/patient_risk.py`**

```python
"""Logica de negocio para la pagina de Riesgo de Pacientes."""
import pandas as pd

from utils.preprocessing import FEATURE_COLS_CANCER, engineer_health_features


def build_risk_table(df_cancer: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Aplica el clasificador de riesgo persistido y retorna la tabla ordenada
    con los pacientes de mayor riesgo primero."""
    df_feat, _ = engineer_health_features(df_cancer, cancer_type_encoder=bundle['cancer_type_encoder'])
    X = df_feat[FEATURE_COLS_CANCER]
    pred_encoded = bundle['model'].predict(X)
    df_feat['Risk_Level_Predicho'] = bundle['target_encoder'].inverse_transform(pred_encoded)

    orden = pd.Categorical(df_feat['Risk_Level_Predicho'],
                            categories=['High', 'Medium', 'Low'], ordered=True)
    df_feat = df_feat.assign(_orden=orden).sort_values('_orden').drop(columns='_orden')

    return df_feat[['Patient_ID', 'Cancer_Type', 'Age', 'Gender', 'Risk_Level_Predicho']]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest codigo/tests/test_patient_risk.py -v`
Expected: PASS (1 test)

---

### Task 7: `codigo/app.py` — Streamlit entrypoint

**Files:**
- Create: `codigo/app.py`
- Test: `codigo/tests/test_app.py`

**Interfaces:**
- Consumes: nothing beyond checking `modelos/metricas.json` exists.
- Produces: the Streamlit entrypoint that pages 1-4 (Tasks 8-11) live alongside as `codigo/pages/*.py` (Streamlit's own multipage convention auto-discovers them — no explicit registration needed).

- [ ] **Step 1: Write `codigo/app.py`**

```python
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ALDIMI Predict — Dashboard", layout="wide")

st.title("ALDIMI Predict — Dashboard de Decisiones")
st.markdown("""
Selecciona una página en la barra lateral:

- **Alertas de Stock**: predicción de stock crítico a 7 y 14 días por insumo.
- **Riesgo de Pacientes**: clasificación de prioridad de atención (Bajo/Medio/Alto).
- **Interpretabilidad**: variables que más influyen en cada modelo.
- **Métricas del Modelo**: F1-macro, MAE, R² y matriz de confusión, para la sustentación.
""")

MODELOS_DIR = Path(__file__).resolve().parent.parent / 'modelos'
if not (MODELOS_DIR / 'metricas.json').exists():
    st.error(
        "No se encontraron modelos entrenados. Ejecuta primero:\n\n"
        "`python codigo/train_models.py`"
    )
```

- [ ] **Step 2: Write the smoke test**

```python
# codigo/tests/test_app.py
from pathlib import Path

from streamlit.testing.v1 import AppTest

CODIGO_DIR = Path(__file__).resolve().parents[1]


def test_app_runs_without_exception():
    at = AppTest.from_file(str(CODIGO_DIR / 'app.py'))
    at.run(timeout=30)
    assert not at.exception
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest codigo/tests/test_app.py -v`
Expected: PASS (1 test) — passes whether or not `modelos/` exists yet, since `app.py` handles the missing-models case with `st.error` instead of raising.

---

### Task 8: `codigo/pages/1_Alertas_Stock.py`

**Files:**
- Create: `codigo/pages/1_Alertas_Stock.py`
- Test: `codigo/tests/test_page_alertas_stock.py`

**Interfaces:**
- Consumes: `utils.model_io.load_bundle` (Task 4), `utils.stock_alerts.build_alert_table` (Task 5).

- [ ] **Step 1: Write `codigo/pages/1_Alertas_Stock.py`**

```python
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_bundle
from utils.stock_alerts import build_alert_table

REPO_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Alertas de Stock", layout="wide")
st.title("Alertas de Stock — Predicción 7 y 14 días")


@st.cache_resource
def _load_models():
    return load_bundle('stock_7d'), load_bundle('stock_14d')


@st.cache_data
def _load_inventory():
    return pd.read_csv(REPO_ROOT / 'inventory_data.csv')


bundle_7, bundle_14 = _load_models()
df_inv = _load_inventory()
alert_table = build_alert_table(df_inv, bundle_7, bundle_14)

tipos = ["Todos"] + sorted(df_inv['Item_Type'].unique().tolist())
tipo_sel = st.selectbox("Filtrar por tipo de insumo", tipos)
if tipo_sel != "Todos":
    alert_table = alert_table[alert_table['Item_Type'] == tipo_sel]

st.dataframe(alert_table, use_container_width=True)

item_sel = st.selectbox("Ver evolución histórica de:", sorted(df_inv['Item_Name'].unique()))
df_item = df_inv[df_inv['Item_Name'] == item_sel].sort_values('Date')
fig = px.line(df_item, x='Date', y='Current_Stock', title=f"Stock histórico — {item_sel}")
fig.add_hline(y=df_item['Min_Required'].mean(), line_dash="dash", line_color="red",
              annotation_text="Mínimo requerido")
st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Write the smoke test**

```python
# codigo/tests/test_page_alertas_stock.py
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

CODIGO_DIR = Path(__file__).resolve().parents[1]
MODELOS_DIR = CODIGO_DIR.parent / 'modelos'

pytestmark = pytest.mark.skipif(
    not (MODELOS_DIR / 'stock_7d.pkl').exists(),
    reason="Requiere ejecutar antes: python codigo/train_models.py",
)


def test_page_runs_without_exception():
    at = AppTest.from_file(str(CODIGO_DIR / 'pages' / '1_Alertas_Stock.py'))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.dataframe) >= 1
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest codigo/tests/test_page_alertas_stock.py -v`
Expected: PASS (1 test)

---

### Task 9: `codigo/pages/2_Riesgo_Pacientes.py`

**Files:**
- Create: `codigo/pages/2_Riesgo_Pacientes.py`
- Test: `codigo/tests/test_page_riesgo_pacientes.py`

**Interfaces:**
- Consumes: `utils.model_io.load_bundle` (Task 4), `utils.patient_risk.build_risk_table` (Task 6).

- [ ] **Step 1: Write `codigo/pages/2_Riesgo_Pacientes.py`**

```python
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_bundle
from utils.patient_risk import build_risk_table

REPO_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Riesgo de Pacientes", layout="wide")
st.title("Riesgo de Pacientes — Nivel de Prioridad de Atención")


@st.cache_resource
def _load_model():
    return load_bundle('riesgo_paciente')


@st.cache_data
def _load_patients():
    return pd.read_csv(REPO_ROOT / 'cancer-risk-factors.csv')


bundle = _load_model()
df_cancer = _load_patients()
risk_table = build_risk_table(df_cancer, bundle)

col1, col2, col3 = st.columns(3)
with col1:
    tipos = ["Todos"] + sorted(risk_table['Cancer_Type'].unique().tolist())
    tipo_sel = st.selectbox("Tipo de cáncer", tipos)
with col2:
    edad_min, edad_max = st.slider(
        "Rango de edad",
        int(risk_table['Age'].min()), int(risk_table['Age'].max()),
        (int(risk_table['Age'].min()), int(risk_table['Age'].max())),
    )
with col3:
    id_buscado = st.text_input("Buscar Patient_ID")

filtrado = risk_table.copy()
if tipo_sel != "Todos":
    filtrado = filtrado[filtrado['Cancer_Type'] == tipo_sel]
filtrado = filtrado[(filtrado['Age'] >= edad_min) & (filtrado['Age'] <= edad_max)]
if id_buscado:
    filtrado = filtrado[filtrado['Patient_ID'].str.contains(id_buscado, case=False, na=False)]

n_high = (filtrado['Risk_Level_Predicho'] == 'High').sum()
st.metric("Pacientes de alto riesgo en la vista actual", n_high)
st.dataframe(filtrado, use_container_width=True)
```

- [ ] **Step 2: Write the smoke test**

```python
# codigo/tests/test_page_riesgo_pacientes.py
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

CODIGO_DIR = Path(__file__).resolve().parents[1]
MODELOS_DIR = CODIGO_DIR.parent / 'modelos'

pytestmark = pytest.mark.skipif(
    not (MODELOS_DIR / 'riesgo_paciente.pkl').exists(),
    reason="Requiere ejecutar antes: python codigo/train_models.py",
)


def test_page_runs_without_exception():
    at = AppTest.from_file(str(CODIGO_DIR / 'pages' / '2_Riesgo_Pacientes.py'))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.dataframe) >= 1
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest codigo/tests/test_page_riesgo_pacientes.py -v`
Expected: PASS (1 test)

---

### Task 10: `codigo/pages/3_Interpretabilidad.py`

**Files:**
- Create: `codigo/pages/3_Interpretabilidad.py`
- Test: `codigo/tests/test_page_interpretabilidad.py`

**Interfaces:**
- Consumes: `utils.model_io.load_metrics` (Task 4).

- [ ] **Step 1: Write `codigo/pages/3_Interpretabilidad.py`**

```python
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_metrics

st.set_page_config(page_title="Interpretabilidad", layout="wide")
st.title("Interpretabilidad — Importancia de Features")

metrics = load_metrics()


def _plot_importances(container, title, importances):
    container.subheader(title)
    if not importances:
        container.info("El modelo ganador no expone feature_importances_.")
        return
    df_imp = pd.Series(importances).sort_values(ascending=True).tail(10).reset_index()
    df_imp.columns = ['Feature', 'Importancia']
    fig = px.bar(df_imp, x='Importancia', y='Feature', orientation='h')
    container.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2)
_plot_importances(col1, "Frente de Salud — Riesgo de Pacientes",
                   metrics['riesgo_paciente'].get('feature_importances'))
_plot_importances(col2, "Frente Logístico — Stock 7 días",
                   metrics['stock_7d'].get('feature_importances'))
```

- [ ] **Step 2: Write the smoke test**

```python
# codigo/tests/test_page_interpretabilidad.py
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

CODIGO_DIR = Path(__file__).resolve().parents[1]
MODELOS_DIR = CODIGO_DIR.parent / 'modelos'

pytestmark = pytest.mark.skipif(
    not (MODELOS_DIR / 'metricas.json').exists(),
    reason="Requiere ejecutar antes: python codigo/train_models.py",
)


def test_page_runs_without_exception():
    at = AppTest.from_file(str(CODIGO_DIR / 'pages' / '3_Interpretabilidad.py'))
    at.run(timeout=60)
    assert not at.exception
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest codigo/tests/test_page_interpretabilidad.py -v`
Expected: PASS (1 test)

---

### Task 11: `codigo/pages/4_Metricas_Modelo.py`

**Files:**
- Create: `codigo/pages/4_Metricas_Modelo.py`
- Test: `codigo/tests/test_page_metricas_modelo.py`

**Interfaces:**
- Consumes: `utils.model_io.load_metrics` (Task 4).

- [ ] **Step 1: Write `codigo/pages/4_Metricas_Modelo.py`**

```python
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_metrics

st.set_page_config(page_title="Métricas del Modelo", layout="wide")
st.title("Métricas del Modelo — Para la Sustentación")

metrics = load_metrics()

st.header("Frente de Salud — Clasificación de Riesgo")
m = metrics['riesgo_paciente']
c1, c2, c3 = st.columns(3)
c1.metric("Modelo ganador", m['modelo_ganador'])
c2.metric("F1-macro", f"{m['f1_macro']:.4f}")
c3.metric("F1-High", f"{m['f1_high']:.4f}")

st.caption(
    f"F1-High = {m['f1_high']:.2f}: de cada 10 pacientes de alto riesgo reales, "
    f"detectamos aproximadamente {round(m['f1_high'] * 10)}."
)
st.write("Matriz de confusión (filas=real, columnas=predicho):")
st.dataframe(pd.DataFrame(m['matriz_confusion'], index=m['clases'], columns=m['clases']))

st.header("Frente Logístico — Predicción de Stock")
for horizon in (7, 14):
    m = metrics[f'stock_{horizon}d']
    st.subheader(f"Horizonte: {horizon} días")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo ganador", m['modelo_ganador'])
    c2.metric("MAE", f"{m['mae']:.1f} unidades")
    c3.metric("R²", f"{m['r2']:.4f}")
```

- [ ] **Step 2: Write the smoke test**

```python
# codigo/tests/test_page_metricas_modelo.py
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

CODIGO_DIR = Path(__file__).resolve().parents[1]
MODELOS_DIR = CODIGO_DIR.parent / 'modelos'

pytestmark = pytest.mark.skipif(
    not (MODELOS_DIR / 'metricas.json').exists(),
    reason="Requiere ejecutar antes: python codigo/train_models.py",
)


def test_page_runs_without_exception():
    at = AppTest.from_file(str(CODIGO_DIR / 'pages' / '4_Metricas_Modelo.py'))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.metric) >= 6  # 3 metricas salud + 3 por cada horizonte logistico
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest codigo/tests/test_page_metricas_modelo.py -v`
Expected: PASS (1 test)

---

### Task 12: Full test suite + manual browser verification

**Files:** none created — verification only.

- [ ] **Step 1: Run the entire test suite**

Run: `pytest codigo/tests -v`
Expected: all tests PASS (preprocessing: 5, train_models: 3, model_io: 2, stock_alerts: 2, patient_risk: 1, app: 1, 4 page smoke tests: 4 — 18 total)

- [ ] **Step 2: Launch the dashboard locally**

Run: `streamlit run codigo/app.py`
Expected: browser opens at `http://localhost:8501` showing the landing page with the 4-page sidebar navigation.

- [ ] **Step 3: Manually verify each page in the browser**

Per this project's rule for UI changes — actually drive the feature in a browser, don't just trust tests:
- **Alertas de Stock:** table renders, filtering by `Item_Type` narrows rows, selecting an item updates the line chart, at least one row shows "Crítico" or confirm none are currently critical.
- **Riesgo de Pacientes:** table renders sorted with `High` first, the `Cancer_Type` filter and age slider narrow the table, the `Patient_ID` search box filters correctly.
- **Interpretabilidad:** both bar charts render with 10 features each.
- **Métricas del Modelo:** all 3+3 metric tiles show numeric values (not `NaN`/blank), confusion matrix table renders with `High`/`Medium`/`Low` labels.

- [ ] **Step 4: Report back**

Summarize which pages were visually confirmed working and flag anything that looked wrong (empty tables, `NaN` metrics, exceptions in the Streamlit UI) instead of claiming the dashboard is done without having looked at it.
