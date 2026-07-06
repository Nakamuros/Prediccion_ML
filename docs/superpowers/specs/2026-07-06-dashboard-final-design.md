# Diseño: Dashboard Final — ALDIMI Predict (Hito 4)

**Fecha:** 2026-07-06
**Sub-proyecto de:** Hito 4 — Trabajo Final (Semana 15)
**Alcance de este documento:** solo el Dashboard Final. El Informe Final, el Video de Impacto y la Sustentación se diseñan y ejecutan después, apoyándose en los artefactos que produzca este dashboard (capturas, métricas persistidas, flujo de demo).

## Contexto

El curso 1ACC0057 exige para el Hito 4 (ver `TP-TF-1ASI0404-Enunciado - 2026-10.pdf`, sección 7):
- Dashboard Final: herramienta de decisión para ALDIMI que muestre alertas de stock y niveles de riesgo de pacientes.
- Video de Impacto (flujo end-to-end).
- Sustentación con análisis de precisión final y recomendaciones éticas.

Estado actual del repo (verificado 2026-07-06):
- Hitos 1-3 completados como notebooks (`Hito 1.ipynb`, `Hito 2.ipynb`, `Hito 3.ipynb`). Sin modelos persistidos (sin `.pkl`/`.joblib`); todo se re-entrena dentro del notebook.
- No existe ninguna app de dashboard interactiva; los "dashboards" previos son imágenes estáticas de matplotlib incrustadas en el notebook.
- No hay `requirements.txt` ni la estructura `/codigo /datos /docs` que exige la Sección 10 del enunciado para el entregable comprimido final.
- La integración con la base de datos común del equipo de IA (mencionada en el enunciado como "Confluencia con el Ecosistema") **no se concretó** en Hito 3 y sigue sin concretarse para Hito 4 — se documentará como limitación/lección aprendida en el informe final, no se bloqueará el dashboard por esto.

Datasets relevantes y columnas confirmadas:
- `inventory_data.csv`: `Item_ID, Item_Type, Item_Name, Current_Stock, Min_Required, Max_Capacity, Unit_Cost, Avg_Usage_Per_Day, Restock_Lead_Time, Vendor_ID` (+ `Date`). `Min_Required` es la base del punto de reorden para las alertas de stock.
- `cancer-risk-factors.csv`: features clínicas/sociales + `Risk_Level` (Low/Medium/High), target del clasificador de riesgo.
- `patient_data.csv`: datos operativos (admisión/alta, insumos usados) — no se usa en este dashboard.

## Decisiones de diseño

1. **Framework:** Streamlit (Python puro, reutiliza código de los notebooks, coincide con el stack sugerido en el enunciado, cero fricción para levantar localmente el día de la sustentación).
2. **Despliegue:** solo local (`streamlit run codigo/app.py`) durante la exposición semana 15, + video de respaldo grabado previamente (recomendado explícitamente por el enunciado, sección 11.3).
3. **Persistencia de modelos:** separación entrenamiento/inferencia. Un script `train_models.py` entrena los modelos ganadores de Hito 3 una vez y los persiste con `joblib`; el dashboard solo carga y predice (arranque instantáneo, buena práctica MLOps citable en el informe).
4. **Interpretabilidad:** se reutiliza `feature_importances_` ya calculado en Hito 3 (sección 5.6 del notebook) — no se introduce SHAP como dependencia nueva a dos semanas de la entrega.

## Estructura de carpetas

```
Prediccion_ML/
├── codigo/
│   ├── train_models.py      # entrena y persiste los 3 modelos finales (Hito 3)
│   ├── app.py                # entrypoint Streamlit (multipágina vía sidebar)
│   ├── pages/
│   │   ├── 1_Alertas_Stock.py
│   │   ├── 2_Riesgo_Pacientes.py
│   │   ├── 3_Interpretabilidad.py
│   │   └── 4_Metricas_Modelo.py
│   └── utils/
│       └── preprocessing.py  # funciones compartidas (reutilizadas de Hito 2/3)
├── datos/                     # csv existentes, sin cambios
├── modelos/                   # .pkl generados por train_models.py (gitignored, se regeneran)
├── docs/
│   └── informe_final.pdf      # Hito 4 (pendiente, próxima fase)
└── requirements.txt           # nuevo — streamlit, scikit-learn, xgboost, joblib, plotly
```

Esta estructura mapea 1:1 con la Sección 10 del enunciado (`/codigo`, `/datos`, `/docs`) para el zip final `TF_1ASI404_NRC_GRUPO_##.zip`.

## Contrato de `train_models.py`

```python
def train_risk_classifier(df_cancer: pd.DataFrame) -> tuple[Pipeline, dict]:
    """SMOTE + mejor modelo (RF o XGBoost, el que ganó en Hito 3) sobre cancer-risk-factors.csv.
    Retorna (pipeline_entrenado, metricas_dict) con F1-macro, F1-High,
    matriz de confusión y feature_importances_.
    """

def train_stock_regressor(df_inventory: pd.DataFrame, horizon: int) -> tuple[Pipeline, dict]:
    """Split temporal + lag features (Stock_Lag1/Lag7/Rolling7) -> mejor regresor.
    horizon: 7 o 14. Retorna (pipeline_entrenado, metricas_dict) con MAE, R², residuos.
    """
```

Al ejecutarse, guarda:
- `modelos/riesgo_paciente.pkl`, `modelos/stock_7d.pkl`, `modelos/stock_14d.pkl` (vía `joblib.dump`)
- `modelos/metricas.json` con las métricas de evaluación de cada modelo, calculadas en el momento del entrenamiento (no se recalculan en el dashboard).

## Las 4 páginas del Dashboard

**1. Alertas de Stock (7/14 días)**
Tabla por `Item_Name`: Stock Actual, Predicción 7d, Predicción 14d, `Min_Required`, y semáforo:
- 🔴 Crítico: predicción < `Min_Required`
- 🟡 Atención: predicción entre `Min_Required` y `Min_Required` × 1.3
- 🟢 OK: por encima de ese margen

Gráfico de línea (histórico + banda de predicción) al seleccionar un ítem. Filtros por `Item_Type` / `Vendor_ID`.

**2. Riesgo de Pacientes**
Tabla de `cancer-risk-factors.csv` con el `Risk_Level` **predicho** (no el real), ordenada con `High` primero. Filtros por `Cancer_Type`, rango de `Age`, buscador por `Patient_ID`. Mismo esquema de color rojo/ámbar/verde que en stock, para consistencia visual entre páginas.

**3. Interpretabilidad**
Gráfico de barras horizontal, top-10 variables por `feature_importances_`, un panel por frente (salud / logística).

**4. Métricas del Modelo**
Lee `modelos/metricas.json`. Muestra F1-macro, F1-High, matriz de confusión (salud) y MAE/R² a 7 y 14 días (logística), con frases de contexto tipo *"F1-High = X: de cada 10 pacientes de alto riesgo, detectamos Y"* para apoyar la ronda de preguntas técnicas de la sustentación (15 min exposición + 5 min Q&A).

## Fuera de alcance de este documento

- Informe Final completo (6 secciones fijas de la rúbrica, sección 9 del enunciado).
- Guion/storyboard del Video de Impacto end-to-end.
- Slides y preparación de la Sustentación (análisis de precisión final + ética).

Estos tres se diseñan en sub-proyectos siguientes, reutilizando las capturas, métricas y flujo de este dashboard.
