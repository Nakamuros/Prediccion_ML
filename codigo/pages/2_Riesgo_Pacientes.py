import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_bundle
from utils.patient_risk import build_risk_table

REPO_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Riesgo de Pacientes", layout="wide")
st.title("Riesgo de Pacientes — Nivel de Prioridad de Atención")


@st.cache_resource
def _load_model():
    return load_bundle('riesgo_paciente')


@st.cache_data
def _load_patients():
    return pd.read_csv(REPO_ROOT / 'cancer-risk-factors.csv')


bundle = _load_model()
df_cancer = _load_patients()
risk_table = build_risk_table(df_cancer, bundle)

col1, col2, col3 = st.columns(3)
with col1:
    tipos = ["Todos"] + sorted(risk_table['Cancer_Type'].unique().tolist())
    tipo_sel = st.selectbox("Tipo de cáncer", tipos)
with col2:
    edad_min, edad_max = st.slider(
        "Rango de edad",
        int(risk_table['Age'].min()), int(risk_table['Age'].max()),
        (int(risk_table['Age'].min()), int(risk_table['Age'].max())),
    )
with col3:
    id_buscado = st.text_input("Buscar Patient_ID")

filtrado = risk_table.copy()
if tipo_sel != "Todos":
    filtrado = filtrado[filtrado['Cancer_Type'] == tipo_sel]
filtrado = filtrado[(filtrado['Age'] >= edad_min) & (filtrado['Age'] <= edad_max)]
if id_buscado:
    filtrado = filtrado[filtrado['Patient_ID'].str.contains(id_buscado, case=False, na=False)]

n_high = (filtrado['Risk_Level_Predicho'] == 'High').sum()
st.metric("Pacientes de alto riesgo en la vista actual", n_high)
st.dataframe(filtrado, use_container_width=True)
