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
