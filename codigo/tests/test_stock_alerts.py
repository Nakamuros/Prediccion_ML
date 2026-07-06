# codigo/tests/test_stock_alerts.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.stock_alerts import build_alert_table, semaforo


class _FixedPredictor:
    """Estimador falso que siempre predice el mismo valor, para aislar la
    logica de semaforo/tabla de la logica real de ML (probada en Task 3)."""
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


def test_semaforo_thresholds():
    assert semaforo(pred=50, min_required=100) == "Crítico"
    assert semaforo(pred=120, min_required=100) == "Atención"
    assert semaforo(pred=140, min_required=100) == "OK"


def test_build_alert_table_uses_most_recent_row_per_item_and_flags_states():
    df_inv = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Item_Name': ['Guantes', 'Guantes'],
        'Item_Type': ['Insumo Medico', 'Insumo Medico'],
        'Current_Stock': [500, 400],
        'Min_Required': [300, 300],
        'Max_Capacity': [1000, 1000],
        'Unit_Cost': [1.0, 1.0],
        'Avg_Usage_Per_Day': [20, 20],
        'Restock_Lead_Time': [5, 5],
        'Vendor_ID': ['V001', 'V001'],
    })
    item_name_encoder = LabelEncoder().fit(df_inv['Item_Name'])
    item_type_encoder = LabelEncoder().fit(df_inv['Item_Type'])

    bundle_7 = {'model': _FixedPredictor(250), 'item_name_encoder': item_name_encoder,
                'item_type_encoder': item_type_encoder}
    bundle_14 = {'model': _FixedPredictor(350), 'item_name_encoder': item_name_encoder,
                 'item_type_encoder': item_type_encoder}

    result = build_alert_table(df_inv, bundle_7, bundle_14)

    assert len(result) == 1  # un registro por item (el mas reciente)
    row = result.iloc[0]
    assert row['Current_Stock'] == 400  # fila del 2024-01-02, no la del 01
    assert row['Estado_7d'] == "Crítico"    # 250 < 300
    assert row['Estado_14d'] == "Atención"  # 300 <= 350 < 390
