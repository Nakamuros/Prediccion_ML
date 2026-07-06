import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_metrics
from utils.ui_style import inject_css
from utils.interpretability_widget import render_feature_importance

inject_css()

st.set_page_config(page_title="Interpretabilidad", layout="wide")
st.title("Interpretabilidad — Importancia de Features")

metrics = load_metrics()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Frente de Salud — Riesgo de Pacientes")
    render_feature_importance(st, metrics['riesgo_paciente'].get('feature_importances'))
with col2:
    st.subheader("Frente Logístico — Stock 7 días")
    render_feature_importance(st, metrics['stock_7d'].get('feature_importances'))