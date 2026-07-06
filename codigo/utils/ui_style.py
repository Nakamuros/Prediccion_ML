import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #F9FAFB !important;
        font-weight: 700;
        border-bottom: 3px solid #FF4B4B;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #F1F5F9 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #374151 !important;
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #F9FAFB !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------- Estilo condicional para tablas de estado/riesgo ----------

ESTADO_COLORS = {
    "Crítico": "#7F1D1D",
    "Atención": "#78350F",
    "OK": "#14532D",
    "Alto": "#7F1D1D",
    "Medio": "#78350F",
    "Bajo": "#14532D",
}

ESTADO_TEXT_COLORS = {
    "Crítico": "#FCA5A5",
    "Atención": "#FCD34D",
    "OK": "#86EFAC",
    "Alto": "#FCA5A5",
    "Medio": "#FCD34D",
    "Bajo": "#86EFAC",
}


def resaltar_estado(val):
    """Para usar en df.style.map (o applymap) sobre columnas de estado/riesgo."""
    bg = ESTADO_COLORS.get(val)
    fg = ESTADO_TEXT_COLORS.get(val)
    if bg:
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"
    return ""


def styler_map(styler, func, subset=None):
    """Compatibilidad entre pandas viejo (.applymap) y nuevo (.map),
    para que el proyecto no truene sin importar la versión de pandas
    instalada en cada máquina."""
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)