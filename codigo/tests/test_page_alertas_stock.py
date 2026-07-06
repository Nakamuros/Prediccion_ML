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
