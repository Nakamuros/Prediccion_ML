from pathlib import Path

import pandas as pd

from utils.preprocessing import (
    engineer_health_features,
    engineer_inventory_features,
    FEATURE_COLS_CANCER,
    FEATURE_COLS_INV,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_engineer_health_features_computes_derived_columns():
    df = pd.DataFrame({
        'Cancer_Type': ['Breast', 'Prostate', 'Breast'],
        'Smoking': [1, 0, 1],
        'Alcohol_Use': [0, 1, 1],
        'Air_Pollution': [1, 1, 0],
        'Diet_Salted_Processed': [2, 1, 3],
        'Diet_Red_Meat': [1, 2, 1],
        'Fruit_Veg_Intake': [3, 2, 1],
        'BRCA_Mutation': [0, 1, 0],
        'Family_History': [1, 1, 0],
        'H_Pylori_Infection': [0, 0, 1],
    })

    result, encoder = engineer_health_features(df)

    assert list(result['Risk_Lifestyle']) == [2, 2, 2]
    assert list(result['Diet_Score']) == [0, 1, 3]
    assert list(result['Genetic_Risk']) == [1, 2, 1]
    assert result['Cancer_Type_enc'].tolist() == encoder.transform(df['Cancer_Type']).tolist()


def test_engineer_health_features_reuses_fitted_encoder():
    df_train = pd.DataFrame({'Cancer_Type': ['Breast', 'Prostate']})
    for col in ['Smoking', 'Alcohol_Use', 'Air_Pollution', 'Diet_Salted_Processed',
                'Diet_Red_Meat', 'Fruit_Veg_Intake', 'BRCA_Mutation', 'Family_History',
                'H_Pylori_Infection']:
        df_train[col] = [0, 0]

    _, encoder = engineer_health_features(df_train)
    result_new, encoder_reused = engineer_health_features(df_train.copy(), cancer_type_encoder=encoder)

    assert encoder_reused is encoder
    assert result_new['Cancer_Type_enc'].tolist() == encoder.transform(df_train['Cancer_Type']).tolist()


def test_engineer_inventory_features_computes_derived_columns():
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-08'],
        'Item_Name': ['Guantes', 'Guantes'],
        'Item_Type': ['Insumo Medico', 'Insumo Medico'],
        'Current_Stock': [500, 400],
        'Min_Required': [300, 300],
        'Avg_Usage_Per_Day': [20, 20],
    })

    result, item_name_enc, item_type_enc = engineer_inventory_features(df)

    assert list(result['Month']) == [1, 1]
    assert list(result['Projected_Stock_7d']) == [360, 260]
    assert list(result['Projected_Stock_14d']) == [220, 120]
    assert list(result['Stock_Ratio']) == [round(500 / 300, 3), round(400 / 300, 3)]
    assert 'Stock_Lag1' not in result.columns  # decision: no lag features (matches Hito 3)


def test_feature_cols_present_after_engineering_on_real_data():
    df_cancer = pd.read_csv(REPO_ROOT / 'cancer-risk-factors.csv')
    result, _ = engineer_health_features(df_cancer)
    assert all(col in result.columns for col in FEATURE_COLS_CANCER)

    df_inv = pd.read_csv(REPO_ROOT / 'inventory_data.csv')
    result_inv, _, _ = engineer_inventory_features(df_inv)
    assert all(col in result_inv.columns for col in FEATURE_COLS_INV)
