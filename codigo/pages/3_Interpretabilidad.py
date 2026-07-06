import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_metrics

st.set_page_config(page_title="Interpretabilidad", layout="wide")
st.title("Interpretabilidad — Importancia de Features")

metrics = load_metrics()


def _plot_importances(container, title, importances):
    container.subheader(title)
    if not importances:
        container.info("El modelo ganador no expone feature_importances_.")
        return
    df_imp = pd.Series(importances).sort_values(ascending=True).tail(10).reset_index()
    df_imp.columns = ['Feature', 'Importancia']
    fig = px.bar(df_imp, x='Importancia', y='Feature', orientation='h')
    container.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2)
_plot_importances(col1, "Frente de Salud — Riesgo de Pacientes",
                   metrics['riesgo_paciente'].get('feature_importances'))
_plot_importances(col2, "Frente Logístico — Stock 7 días",
                   metrics['stock_7d'].get('feature_importances'))
