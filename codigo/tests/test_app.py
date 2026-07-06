# codigo/tests/test_app.py
from pathlib import Path

from streamlit.testing.v1 import AppTest

CODIGO_DIR = Path(__file__).resolve().parents[1]


def test_app_runs_without_exception():
    at = AppTest.from_file(str(CODIGO_DIR / 'app.py'))
    at.run(timeout=30)
    assert not at.exception
