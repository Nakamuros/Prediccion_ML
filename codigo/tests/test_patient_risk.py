# codigo/tests/test_patient_risk.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.patient_risk import build_risk_table


class _FixedClassifier:
    def __init__(self, encoded_values):
        self.encoded_values = encoded_values

    def predict(self, X):
        return self.encoded_values[:len(X)]


def test_build_risk_table_orders_high_risk_first():
    df_cancer = pd.DataFrame({
        'Patient_ID': ['P1', 'P2', 'P3'],
        'Cancer_Type': ['Breast', 'Breast', 'Lung'],
        'Age': [40, 55, 60],
        'Gender': [0, 1, 0],
        'Smoking': [0, 1, 1], 'Alcohol_Use': [0, 0, 1], 'Obesity': [0, 1, 0],
        'Family_History': [0, 1, 1], 'Diet_Red_Meat': [1, 2, 3],
        'Diet_Salted_Processed': [1, 2, 1], 'Fruit_Veg_Intake': [3, 1, 1],
        'Physical_Activity': [2, 1, 1], 'Air_Pollution': [1, 1, 1],
        'Occupational_Hazards': [0, 0, 1], 'BRCA_Mutation': [0, 0, 1],
        'H_Pylori_Infection': [0, 0, 0], 'Calcium_Intake': [2, 2, 1],
        'BMI': [22.0, 27.0, 30.0], 'Physical_Activity_Level': [1, 1, 0],
    })

    cancer_type_encoder = LabelEncoder().fit(df_cancer['Cancer_Type'])
    target_encoder = LabelEncoder().fit(['High', 'Low', 'Medium'])
    # target_encoder.classes_ ordenadas alfabeticamente: High=0, Low=1, Medium=2
    bundle = {
        'model': _FixedClassifier([1, 2, 0]),  # P1->Low, P2->Medium, P3->High
        'cancer_type_encoder': cancer_type_encoder,
        'target_encoder': target_encoder,
    }

    result = build_risk_table(df_cancer, bundle)

    assert list(result['Patient_ID']) == ['P3', 'P2', 'P1']
    assert result.iloc[0]['Risk_Level_Predicho'] == 'High'
