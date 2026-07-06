import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_metrics

st.set_page_config(page_title="Métricas del Modelo", layout="wide")
st.title("Métricas del Modelo — Para la Sustentación")

metrics = load_metrics()

st.header("Frente de Salud — Clasificación de Riesgo")
m = metrics['riesgo_paciente']
c1, c2, c3 = st.columns(3)
c1.metric("Modelo ganador", m['modelo_ganador'])
c2.metric("F1-macro", f"{m['f1_macro']:.4f}")
c3.metric("F1-High", f"{m['f1_high']:.4f}")

st.caption(
    f"F1-High = {m['f1_high']:.2f}: de cada 10 pacientes de alto riesgo reales, "
    f"detectamos aproximadamente {round(m['f1_high'] * 10)}."
)
st.write("Matriz de confusión (filas=real, columnas=predicho):")
st.dataframe(pd.DataFrame(m['matriz_confusion'], index=m['clases'], columns=m['clases']))

st.header("Frente Logístico — Predicción de Stock")
for horizon in (7, 14):
    m = metrics[f'stock_{horizon}d']
    st.subheader(f"Horizonte: {horizon} días")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo ganador", m['modelo_ganador'])
    c2.metric("MAE", f"{m['mae']:.1f} unidades")
    c3.metric("R²", f"{m['r2']:.4f}")
