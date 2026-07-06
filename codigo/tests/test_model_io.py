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
