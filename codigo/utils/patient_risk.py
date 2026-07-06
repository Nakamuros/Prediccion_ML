"""Logica de negocio para la pagina de Riesgo de Pacientes."""
import pandas as pd

from utils.preprocessing import FEATURE_COLS_CANCER, engineer_health_features


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

    return df_feat[['Patient_ID', 'Cancer_Type', 'Age', 'Gender', 'Risk_Level_Predicho']]
