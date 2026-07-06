"""Normalización de nombres de columnas y features para presentación en UI."""
import pandas as pd

COLUMN_LABELS = {
    # Alertas de Stock
    'Item_Name': 'Insumo',
    'Item_Type': 'Tipo',
    'Vendor_ID': 'Proveedor',
    'Current_Stock': 'Stock Actual',
    'Min_Required': 'Mínimo Requerido',
    'Pred_7d': 'Predicción 7 días',
    'Estado_7d': 'Estado (7d)',
    'Pred_14d': 'Predicción 14 días',
    'Estado_14d': 'Estado (14d)',

    # Riesgo de Pacientes
    'Patient_ID': 'ID Paciente',
    'Cancer_Type': 'Tipo de Cáncer',
    'Age': 'Edad',
    'Gender': 'Género',
    'Risk_Level_Predicho': 'Nivel de Riesgo',

    # Interpretabilidad
    'Feature': 'Variable',
    'Importancia': 'Importancia',
}

# Traducción específica de nombres de features (usadas en feature_importances_)
FEATURE_LABELS = {
    # --- Frente de Salud ---
    'Risk_Lifestyle': 'Estilo de Vida de Riesgo',
    'Family_History': 'Antecedentes Familiares',
    'Air_Pollution': 'Contaminación del Aire',
    'Diet_Score': 'Puntaje de Dieta',
    'H_Pylori_Infection': 'Infección por H. Pylori',
    'Diet_Salted_Processed': 'Dieta Procesada/Salada',
    'Occupational_Hazards': 'Riesgos Ocupacionales',
    'Alcohol_Use': 'Consumo de Alcohol',
    'Diet_Red_Meat': 'Consumo de Carne Roja',
    'Obesity': 'Obesidad',
    'Smoking': 'Tabaquismo',
    'Fruit_Veg_Intake': 'Consumo de Frutas y Verduras',
    'Physical_Activity': 'Actividad Física',
    'Physical_Activity_Level': 'Nivel de Actividad Física',
    'BRCA_Mutation': 'Mutación BRCA',
    'Calcium_Intake': 'Consumo de Calcio',
    'BMI': 'Índice de Masa Corporal',
    'Genetic_Risk': 'Riesgo Genético',
    'Cancer_Type_enc': 'Tipo de Cáncer (codificado)',

    # --- Frente Logístico ---
    'Days_Until_Stockout': 'Días Hasta Agotar Stock',
    'Current_Stock': 'Stock Actual',
    'Avg_Usage_Per_Day': 'Uso Promedio Diario',
    'Month': 'Mes',
    'Max_Capacity': 'Capacidad Máxima',
    'Day_of_Week': 'Día de la Semana',
    'Restock_Lead_Time': 'Tiempo de Reabastecimiento',
    'Item_Name_enc': 'Insumo (codificado)',
    'Item_Type_enc': 'Tipo de Insumo (codificado)',
    'Unit_Cost': 'Costo Unitario',
    'Min_Required': 'Mínimo Requerido',
    'Stock_Ratio': 'Ratio de Stock',
    'Quarter': 'Trimestre',
}


def pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas de un DataFrame para presentación."""
    df = df.copy()
    nuevas_cols = {col: COLUMN_LABELS.get(col, col.replace('_', ' ').title()) for col in df.columns}
    return df.rename(columns=nuevas_cols)


def pretty_feature_name(nombre: str) -> str:
    """Traduce el nombre de una feature individual a español, con fallback
    automático si no está en el diccionario explícito."""
    if nombre in FEATURE_LABELS:
        return FEATURE_LABELS[nombre]
    if nombre in COLUMN_LABELS:
        return COLUMN_LABELS[nombre]
    return nombre.replace('_', ' ').title()