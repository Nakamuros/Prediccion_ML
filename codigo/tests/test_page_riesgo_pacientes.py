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
