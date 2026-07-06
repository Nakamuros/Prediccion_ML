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

    df_cancer = pd.read_csv(REPO_ROOT / 'datos' / 'cancer-risk-factors.csv')
    df_inv = pd.read_csv(REPO_ROOT / 'datos' / 'inventory_data.csv')

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
