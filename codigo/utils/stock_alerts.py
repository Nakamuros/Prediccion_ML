"""Logica de negocio para la pagina de Alertas de Stock."""
import pandas as pd

from utils.preprocessing import FEATURE_COLS_INV, engineer_inventory_features


def semaforo(pred: float, min_required: float) -> str:
    """Clasifica una prediccion de stock segun el margen sobre Min_Required."""
    if pred < min_required:
        return "Crítico"
    if pred < min_required * 1.3:
        return "Atención"
    return "OK"


def build_alert_table(df_inv: pd.DataFrame, bundle_7: dict, bundle_14: dict) -> pd.DataFrame:
    """Tabla de alertas con predicciones a 7 y 14 dias para el registro mas
    reciente de cada insumo."""
    df_feat, _, _ = engineer_inventory_features(
        df_inv,
        item_name_encoder=bundle_7['item_name_encoder'],
        item_type_encoder=bundle_7['item_type_encoder'],
    )
    latest = (df_feat.sort_values('Date')
              .groupby('Item_Name', as_index=False)
              .last())

    X = latest[FEATURE_COLS_INV]
    latest['Pred_7d'] = bundle_7['model'].predict(X)
    latest['Pred_14d'] = bundle_14['model'].predict(X)

    latest['Pred_7d'] = bundle_7['model'].predict(X).clip(min=0)
    latest['Pred_14d'] = bundle_14['model'].predict(X).clip(min=0)

    latest['Estado_7d'] = latest.apply(lambda r: semaforo(r['Pred_7d'], r['Min_Required']), axis=1)
    latest['Estado_14d'] = latest.apply(lambda r: semaforo(r['Pred_14d'], r['Min_Required']), axis=1)

    return latest[['Item_Name', 'Item_Type', 'Vendor_ID', 'Current_Stock',
                    'Min_Required', 'Pred_7d', 'Estado_7d', 'Pred_14d', 'Estado_14d']]
