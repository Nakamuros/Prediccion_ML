import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_bundle, load_metrics
from utils.stock_alerts import build_alert_table
from utils.ui_style import inject_css, resaltar_estado, styler_map
from utils.labels import pretty_columns
from utils.interpretability_widget import render_feature_importance

inject_css()

REPO_ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Alertas de Stock", layout="wide")
st.title("Alertas de Stock — Predicción 7 y 14 días")


@st.cache_resource
def _load_models():
    return load_bundle('stock_7d'), load_bundle('stock_14d')


@st.cache_data
def _load_inventory():
    return pd.read_csv(REPO_ROOT / 'datos' / 'inventory_data.csv')


bundle_7, bundle_14 = _load_models()
df_inv = _load_inventory()
alert_table = build_alert_table(df_inv, bundle_7, bundle_14)

# ---------- PANEL DE FILTROS ----------
with st.container(border=True):
    st.markdown("##### Filtros")
    f1, f2, f3 = st.columns(3)
    with f1:
        tipos = ["Todos"] + sorted(alert_table['Item_Type'].unique().tolist())
        tipo_sel = st.selectbox("Tipo de insumo", tipos)
    with f2:
        proveedores = ["Todos"] + sorted(alert_table['Vendor_ID'].unique().tolist())
        proveedor_sel = st.selectbox("Proveedor", proveedores)
    with f3:
        estados = ["Todos", "Crítico", "Atención", "OK"]
        estado_sel = st.selectbox("Estado (7 días)", estados)

# ---------- APLICAR FILTROS ----------
filtrado = alert_table.copy()
if tipo_sel != "Todos":
    filtrado = filtrado[filtrado['Item_Type'] == tipo_sel]
if proveedor_sel != "Todos":
    filtrado = filtrado[filtrado['Vendor_ID'] == proveedor_sel]
if estado_sel != "Todos":
    filtrado = filtrado[filtrado['Estado_7d'] == estado_sel]

# Orden por criticidad: Crítico primero, luego Atención, luego OK
orden_estado = pd.Categorical(filtrado['Estado_7d'], categories=['Crítico', 'Atención', 'OK'], ordered=True)
filtrado = filtrado.assign(_orden=orden_estado).sort_values('_orden').drop(columns='_orden')

# ---------- KPIs ----------
n_critico_7d = (filtrado['Estado_7d'] == 'Crítico').sum()
n_atencion_7d = (filtrado['Estado_7d'] == 'Atención').sum()
n_critico_14d = (filtrado['Estado_14d'] == 'Crítico').sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Insumos en la vista", len(filtrado))
kpi2.metric("Críticos (7 días)", n_critico_7d)
kpi3.metric("En atención (7 días)", n_atencion_7d)
kpi4.metric("Críticos (14 días)", n_critico_14d)

st.markdown("###  ")

# ---------- ALERTA VISUAL: SOLO CRÍTICOS ----------
criticos = filtrado[filtrado['Estado_7d'] == 'Crítico']
if not criticos.empty:
    st.warning(f"⚠️ {len(criticos)} insumo(s) en estado Crítico requieren atención inmediata:")
    cols = st.columns(min(len(criticos), 4))
    for col, (_, row) in zip(cols, criticos.head(4).iterrows()):
        with col:
            st.markdown(f"""
            <div style="border-left:4px solid #EF4444; background:#1E1315; padding:10px; border-radius:6px;">
                <strong style="color:#F9FAFB;">{row['Item_Name']}</strong><br>
                <span style="color:#94A3B8; font-size:0.85rem;">Stock: {row['Current_Stock']:.0f} / Mín: {row['Min_Required']:.0f}</span>
            </div>
            """, unsafe_allow_html=True)

# ---------- TABLA ----------
tabla_view = pretty_columns(filtrado)
styled = styler_map(
    tabla_view.style,
    resaltar_estado,
    subset=['Estado (7d)', 'Estado (14d)']
).format({
    'Stock Actual': '{:.0f}',
    'Mínimo Requerido': '{:.0f}',
    'Predicción 7 días': '{:.1f}',
    'Predicción 14 días': '{:.1f}',
})
st.dataframe(styled, width='stretch', hide_index=True)