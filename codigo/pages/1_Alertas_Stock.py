import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_bundle
from utils.stock_alerts import build_alert_table

REPO_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Alertas de Stock", layout="wide")
st.title("Alertas de Stock — Predicción 7 y 14 días")


@st.cache_resource
def _load_models():
    return load_bundle('stock_7d'), load_bundle('stock_14d')


@st.cache_data
def _load_inventory():
    return pd.read_csv(REPO_ROOT / 'inventory_data.csv')


bundle_7, bundle_14 = _load_models()
df_inv = _load_inventory()
alert_table = build_alert_table(df_inv, bundle_7, bundle_14)

tipos = ["Todos"] + sorted(df_inv['Item_Type'].unique().tolist())
tipo_sel = st.selectbox("Filtrar por tipo de insumo", tipos)
if tipo_sel != "Todos":
    alert_table = alert_table[alert_table['Item_Type'] == tipo_sel]

st.dataframe(alert_table, use_container_width=True)

item_sel = st.selectbox("Ver evolución histórica de:", sorted(df_inv['Item_Name'].unique()))
df_item = df_inv[df_inv['Item_Name'] == item_sel].sort_values('Date')
fig = px.line(df_item, x='Date', y='Current_Stock', title=f"Stock histórico — {item_sel}")
fig.add_hline(y=df_item['Min_Required'].mean(), line_dash="dash", line_color="red",
              annotation_text="Mínimo requerido")
st.plotly_chart(fig, use_container_width=True)
