"""Ingenieria de caracteristicas compartida (Hito 2/3) para ambos frentes.

Funciones puras: reciben y devuelven DataFrames, no leen archivos.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS_CANCER = [
    'Age', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity', 'Family_History',
    'Diet_Red_Meat', 'Diet_Salted_Processed', 'Fruit_Veg_Intake',
    'Physical_Activity', 'Air_Pollution', 'Occupational_Hazards',
    'BRCA_Mutation', 'H_Pylori_Infection', 'Calcium_Intake',
    'BMI', 'Physical_Activity_Level', 'Cancer_Type_enc',
    'Risk_Lifestyle', 'Diet_Score', 'Genetic_Risk',
    # Overall_Risk_Score y Metastasis_Status excluidas: data leakage (Hito 2)
]

FEATURE_COLS_INV = [
    'Current_Stock', 'Min_Required', 'Max_Capacity', 'Unit_Cost',
    'Avg_Usage_Per_Day', 'Restock_Lead_Time',
    'Item_Name_enc', 'Item_Type_enc',
    'Month', 'Quarter', 'Day_of_Week',
    'Stock_Ratio', 'Days_Until_Stockout',
]


def engineer_health_features(df_cancer: pd.DataFrame, cancer_type_encoder: LabelEncoder = None):
    """Reproduce la ingenieria de caracteristicas de Hito 2/3 para el frente de salud.

    Si cancer_type_encoder es None se ajusta uno nuevo (modo entrenamiento);
    si se pasa uno ya ajustado se reutiliza (modo inferencia), para garantizar
    la misma codificacion Cancer_Type -> entero entre train y predict.
    """
    df = df_cancer.copy()
    df['Risk_Lifestyle'] = df['Smoking'] + df['Alcohol_Use'] + df['Air_Pollution']
    df['Diet_Score'] = df['Diet_Salted_Processed'] + df['Diet_Red_Meat'] - df['Fruit_Veg_Intake']
    df['Genetic_Risk'] = df['BRCA_Mutation'] + df['Family_History'] + df['H_Pylori_Infection']

    if cancer_type_encoder is None:
        cancer_type_encoder = LabelEncoder()
        df['Cancer_Type_enc'] = cancer_type_encoder.fit_transform(df['Cancer_Type'])
    else:
        df['Cancer_Type_enc'] = cancer_type_encoder.transform(df['Cancer_Type'])

    return df, cancer_type_encoder


def engineer_inventory_features(df_inv: pd.DataFrame, item_name_encoder: LabelEncoder = None,
                                 item_type_encoder: LabelEncoder = None):
    """Reproduce la ingenieria de caracteristicas de Hito 2/3 para el frente logistico.

    No incluye features de lag temporal (Stock_Lag1/7, RollingMean7): en Hito 3
    se calcularon pero el entrenamiento del regresor final uso FEATURE_COLS_INV
    (sin lag), no FEATURE_COLS_INV_TS. Se reproduce tal cual para no alterar las
    metricas ya sustentadas (decision confirmada para Hito 4).
    """
    df = df_inv.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    df['Projected_Stock_7d'] = (df['Current_Stock'] - df['Avg_Usage_Per_Day'] * 7).clip(lower=0)
    df['Projected_Stock_14d'] = (df['Current_Stock'] - df['Avg_Usage_Per_Day'] * 14).clip(lower=0)
    df['Days_Until_Stockout'] = (df['Current_Stock'] / df['Avg_Usage_Per_Day']).round(2)
    df['Stock_Ratio'] = df['Current_Stock'] / df['Min_Required']

    if item_name_encoder is None:
        item_name_encoder = LabelEncoder()
        df['Item_Name_enc'] = item_name_encoder.fit_transform(df['Item_Name'])
    else:
        df['Item_Name_enc'] = item_name_encoder.transform(df['Item_Name'])

    if item_type_encoder is None:
        item_type_encoder = LabelEncoder()
        df['Item_Type_enc'] = item_type_encoder.fit_transform(df['Item_Type'])
    else:
        df['Item_Type_enc'] = item_type_encoder.transform(df['Item_Type'])

    return df, item_name_encoder, item_type_encoder
