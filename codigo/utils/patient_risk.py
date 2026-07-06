"""Logica de negocio para la pagina de Riesgo de Pacientes."""
import pandas as pd

from utils.preprocessing import FEATURE_COLS_CANCER, engineer_health_features

GENDER_MAP = {0: "Femenino", 1: "Masculino"}

RISK_LEVEL_MAP = {
    "High": "Alto",
    "Medium": "Medio",
    "Low": "Bajo",
}

CANCER_TYPE_MAP = {
    "Breast": "Mama",
    "Prostate": "Próstata",
    "Skin": "Piel",
    "Colon": "Colon",
    "Lung": "Pulmón",
}

def build_risk_table(df_cancer: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Aplica el clasificador de riesgo persistido y retorna la tabla ordenada
    con los pacientes de mayor riesgo primero."""
    df_feat, _ = engineer_health_features(df_cancer, cancer_type_encoder=bundle['cancer_type_encoder'])
    X = df_feat[FEATURE_COLS_CANCER]
    pred_encoded = bundle['model'].predict(X)
    df_feat['Risk_Level_Predicho'] = bundle['target_encoder'].inverse_transform(pred_encoded)

    orden = pd.Categorical(df_feat['Risk_Level_Predicho'],
                            categories=['High', 'Medium', 'Low'], ordered=True)
    df_feat = df_feat.assign(_orden=orden).sort_values('_orden').drop(columns='_orden')

    # Traducciones solo para presentación (no afectan el modelo, que ya predijo)
    df_feat['Gender'] = df_feat['Gender'].map(GENDER_MAP).fillna(df_feat['Gender'])
    df_feat['Risk_Level_Predicho'] = df_feat['Risk_Level_Predicho'].map(RISK_LEVEL_MAP).fillna(df_feat['Risk_Level_Predicho'])
    df_feat['Cancer_Type'] = df_feat['Cancer_Type'].map(CANCER_TYPE_MAP).fillna(df_feat['Cancer_Type'])

    return df_feat[['Patient_ID', 'Cancer_Type', 'Age', 'Gender', 'Risk_Level_Predicho']]

def get_patient_detail(df_cancer: pd.DataFrame, bundle: dict, patient_id: str) -> dict:
    """Devuelve todas las variables predictoras (features) de un paciente
    específico, para mostrar en un detalle/dialog. No incluye columnas de
    codificación interna (_enc) ni la variable objetivo."""
    df_feat, _ = engineer_health_features(df_cancer, cancer_type_encoder=bundle['cancer_type_encoder'])
    fila = df_feat[df_feat['Patient_ID'] == patient_id]
    if fila.empty:
        return {}

    fila = fila.iloc[0]
    # Excluye columnas técnicas/codificadas que no aportan al detalle visual
    excluir = {'Cancer_Type_enc', 'Overall_Risk_Score', 'Metastasis_Status'}
    detalle = {
        col: fila[col] for col in FEATURE_COLS_CANCER
        if col not in excluir
    }
    return detalle