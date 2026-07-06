from pathlib import Path

import streamlit as st
from utils.ui_style import inject_css
inject_css()

st.set_page_config(page_title="ALDIMI Predict — Dashboard", layout="wide")

st.title("ALDIMI Predict — Dashboard de Decisiones")
st.markdown("""
Selecciona una página en la barra lateral:

- **Alertas de Stock**: predicción de stock crítico a 7 y 14 días por insumo.
- **Riesgo de Pacientes**: clasificación de prioridad de atención (Bajo/Medio/Alto).
- **Interpretabilidad**: variables que más influyen en cada modelo.
""")

MODELOS_DIR = Path(__file__).resolve().parent.parent / 'modelos'
if not (MODELOS_DIR / 'metricas.json').exists():
    st.error(
        "No se encontraron modelos entrenados. Ejecuta primero:\n\n"
        "`python codigo/train_models.py`"
    )
