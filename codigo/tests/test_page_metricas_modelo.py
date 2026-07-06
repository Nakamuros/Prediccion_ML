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
