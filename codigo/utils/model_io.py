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
