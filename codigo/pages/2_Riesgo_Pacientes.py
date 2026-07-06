import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_io import load_bundle, load_metrics
from utils.patient_risk import build_risk_table, get_patient_detail, predict_new_patient
from utils.ui_style import inject_css, resaltar_estado, styler_map
from utils.labels import pretty_columns, pretty_feature_name
from utils.interpretability_widget import render_feature_importance

inject_css()

REPO_ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Riesgo de Pacientes", layout="wide")
st.title("Riesgo de Pacientes — Nivel de Prioridad de Atención")


@st.cache_resource
def _load_model():
    return load_bundle('riesgo_paciente')


@st.cache_data
def _load_patients():
    return pd.read_csv(REPO_ROOT / 'datos' / 'cancer-risk-factors.csv')


bundle = _load_model()
df_cancer = _load_patients()
risk_table = build_risk_table(df_cancer, bundle)

# Combina pacientes registrados manualmente en esta sesión
if 'pacientes_nuevos' in st.session_state and st.session_state['pacientes_nuevos']:
    df_nuevos = pd.DataFrame(st.session_state['pacientes_nuevos'])
    risk_table = pd.concat([df_nuevos, risk_table], ignore_index=True)

# ---------- DIÁLOGO DE DETALLE ----------
@st.dialog("Detalle del paciente", width="large")
def mostrar_detalle_paciente(patient_id, cancer_type, edad, genero, riesgo):
    st.markdown(f"### {patient_id}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Edad", edad)
    c2.metric("Género", genero)
    c3.metric("Nivel de Riesgo", riesgo)
    st.caption(f"Tipo de cáncer: {cancer_type}")
    st.divider()

    st.markdown("##### Datos del paciente")
    detalle = get_patient_detail(df_cancer, bundle, patient_id)
    if not detalle:
        st.warning("No se encontró información adicional para este paciente.")
        return

    df_detalle = pd.DataFrame([
        {"Variable": pretty_feature_name(k), "Valor": v}
        for k, v in detalle.items()
    ])
    st.dataframe(df_detalle, width='stretch', hide_index=True)

# ---------- DIÁLOGO: NUEVO PACIENTE ----------
@st.dialog("Registrar nuevo paciente", width="large")
def formulario_nuevo_paciente():
    with st.form("form_nuevo_paciente"):
        st.markdown("##### Datos generales")
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_id = st.text_input("ID Paciente", placeholder="Ej. NP0001")
            edad = st.number_input("Edad", min_value=18, max_value=100, value=50)
        with c2:
            genero = st.selectbox("Género", ["Femenino", "Masculino"])
            cancer_type = st.selectbox("Tipo de cáncer", sorted(df_cancer['Cancer_Type'].unique()))
        with c3:
            bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            actividad_nivel = st.slider("Nivel de Actividad Física", 0, 10, 5)

        st.divider()
        st.markdown("##### Factores de estilo de vida (escala 0-10)")
        c4, c5, c6 = st.columns(3)
        with c4:
            smoking = st.slider("Tabaquismo", 0, 10, 0)
            alcohol = st.slider("Consumo de Alcohol", 0, 10, 0)
        with c5:
            obesity = st.slider("Obesidad", 0, 10, 0)
            actividad = st.slider("Actividad Física", 0, 10, 5)
        with c6:
            aire = st.slider("Contaminación del Aire", 0, 10, 0)
            ocupacional = st.slider("Riesgos Ocupacionales", 0, 10, 0)

        st.divider()
        st.markdown("##### Dieta")
        c7, c8, c9 = st.columns(3)
        with c7:
            carne_roja = st.slider("Consumo de Carne Roja", 0, 10, 0)
        with c8:
            procesados = st.slider("Dieta Procesada/Salada", 0, 10, 0)
        with c9:
            frutas_verduras = st.slider("Consumo de Frutas y Verduras", 0, 10, 5)

        st.divider()
        st.markdown("##### Antecedentes clínicos")
        c10, c11, c12, c13 = st.columns(4)
        with c10:
            antecedentes = st.selectbox("Antecedentes Familiares", ["No", "Sí"])
        with c11:
            brca = st.selectbox("Mutación BRCA", ["No", "Sí"])
        with c12:
            h_pylori = st.selectbox("Infección H. Pylori", ["No", "Sí"])
        with c13:
            calcio = st.slider("Consumo de Calcio", 0, 10, 5)

        enviado = st.form_submit_button("Predecir riesgo", width='stretch')

    if enviado:
        if not patient_id:
            st.error("Por favor ingresa un ID de paciente.")
            return

        datos = {
            'Patient_ID': patient_id,
            'Cancer_Type': cancer_type,
            'Age': edad,
            'Gender': 1 if genero == "Masculino" else 0,
            'Smoking': smoking,
            'Alcohol_Use': alcohol,
            'Obesity': obesity,
            'Family_History': 1 if antecedentes == "Sí" else 0,
            'Diet_Red_Meat': carne_roja,
            'Diet_Salted_Processed': procesados,
            'Fruit_Veg_Intake': frutas_verduras,
            'Physical_Activity': actividad,
            'Air_Pollution': aire,
            'Occupational_Hazards': ocupacional,
            'BRCA_Mutation': 1 if brca == "Sí" else 0,
            'H_Pylori_Infection': 1 if h_pylori == "Sí" else 0,
            'Calcium_Intake': calcio,
            'BMI': bmi,
            'Physical_Activity_Level': actividad_nivel,
        }

        resultado = predict_new_patient(bundle, datos)
        riesgo = resultado["riesgo"]

        color_riesgo = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}.get(riesgo, "⚪")
        st.success(f"Predicción completada para **{patient_id}**")
        st.markdown(f"### {color_riesgo} Nivel de Riesgo Predicho: **{riesgo}**")

        # Guarda en sesión para que aparezca en la tabla/tarjetas de esta sesión
        if 'pacientes_nuevos' not in st.session_state:
            st.session_state['pacientes_nuevos'] = []
        st.session_state['pacientes_nuevos'].append({
            'Patient_ID': patient_id,
            'Cancer_Type': cancer_type,
            'Age': edad,
            'Gender': genero,
            'Risk_Level_Predicho': riesgo,
        })

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("➕ Registrar nuevo paciente", width='stretch'):
        formulario_nuevo_paciente()

# ---------- PANEL DE FILTROS ----------
with st.container(border=True):
    st.markdown("##### Filtros")
    fila1_col1, fila1_col2, fila1_col3, fila1_col4 = st.columns(4)
    with fila1_col1:
        tipos = ["Todos"] + sorted(risk_table['Cancer_Type'].unique().tolist())
        tipo_sel = st.selectbox("Tipo de cáncer", tipos)
    with fila1_col2:
        generos = ["Todos"] + sorted(risk_table['Gender'].unique().tolist())
        genero_sel = st.selectbox("Género", generos)
    with fila1_col3:
        niveles = ["Todos", "Alto", "Medio", "Bajo"]
        nivel_sel = st.selectbox("Nivel de riesgo", niveles)
    with fila1_col4:
        id_buscado = st.text_input("Buscar ID Paciente")

    edad_min, edad_max = st.slider(
        "Rango de edad",
        int(risk_table['Age'].min()), int(risk_table['Age'].max()),
        (int(risk_table['Age'].min()), int(risk_table['Age'].max())),
    )

# ---------- APLICAR FILTROS ----------
filtrado = risk_table.copy()
if tipo_sel != "Todos":
    filtrado = filtrado[filtrado['Cancer_Type'] == tipo_sel]
if genero_sel != "Todos":
    filtrado = filtrado[filtrado['Gender'] == genero_sel]
if nivel_sel != "Todos":
    filtrado = filtrado[filtrado['Risk_Level_Predicho'] == nivel_sel]
filtrado = filtrado[(filtrado['Age'] >= edad_min) & (filtrado['Age'] <= edad_max)]
if id_buscado:
    filtrado = filtrado[filtrado['Patient_ID'].str.contains(id_buscado, case=False, na=False)]

# ---------- KPIs ----------
n_alto = (filtrado['Risk_Level_Predicho'] == 'Alto').sum()
n_medio = (filtrado['Risk_Level_Predicho'] == 'Medio').sum()
n_bajo = (filtrado['Risk_Level_Predicho'] == 'Bajo').sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total en la vista", len(filtrado))
kpi2.metric("Alto riesgo", n_alto)
kpi3.metric("Riesgo medio", n_medio)
kpi4.metric("Bajo riesgo", n_bajo)


# ---------- SELECTOR DE VISTA ----------
vista = st.radio("Mostrar como:", ["Tabla", "Tarjetas"], horizontal=True)

if vista == "Tabla":
    tabla_view = pretty_columns(filtrado)
    styled = styler_map(tabla_view.style, resaltar_estado, subset=['Nivel de Riesgo'])
    st.dataframe(styled, width='stretch', hide_index=True)

else:
    if filtrado.empty:
        st.info("No hay pacientes que coincidan con los filtros.")
    else:
        BORDE_COLOR = {"Alto": "#EF4444", "Medio": "#F59E0B", "Bajo": "#22C55E"}
        ICONO_RIESGO = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}

        filas = [filtrado.iloc[i:i+3] for i in range(0, len(filtrado), 3)]
        for fila in filas:
            cols = st.columns(3)
            for col, (_, paciente) in zip(cols, fila.iterrows()):
                color = BORDE_COLOR.get(paciente['Risk_Level_Predicho'], "#374151")
                icono = ICONO_RIESGO.get(paciente['Risk_Level_Predicho'], "⚪")
                with col:
                    st.markdown(f"""
                    <div style="
                        border: 1px solid {color};
                        border-left: 5px solid {color};
                        border-radius: 10px 10px 0 0;
                        padding: 16px;
                        background-color: #111827;
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="font-size:1.1rem; color:#F9FAFB;">{paciente['Patient_ID']}</strong>
                            <span style="font-size:1.3rem;">{icono}</span>
                        </div>
                        <hr style="border-color:#374151; margin:8px 0;">
                        <p style="margin:4px 0; color:#94A3B8;">Tipo de cáncer: <strong style="color:#F1F5F9;">{paciente['Cancer_Type']}</strong></p>
                        <p style="margin:4px 0; color:#94A3B8;">Edad: <strong style="color:#F1F5F9;">{paciente['Age']}</strong> &nbsp;|&nbsp; Género: <strong style="color:#F1F5F9;">{paciente['Gender']}</strong></p>
                        <p style="margin:8px 0 0 0; color:{color}; font-weight:700;">Riesgo: {paciente['Risk_Level_Predicho']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Ver detalle", key=f"detalle_{paciente['Patient_ID']}", width='stretch'):
                        mostrar_detalle_paciente(
                            paciente['Patient_ID'],
                            paciente['Cancer_Type'],
                            paciente['Age'],
                            paciente['Gender'],
                            paciente['Risk_Level_Predicho'],
                        )
