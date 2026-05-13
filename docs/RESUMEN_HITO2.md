# Resumen Técnico — Hito 2: ALDIMI Predict
### Preparación de Datos y Modelo Baseline | CRISP-DM Fases 3 y 4

> Documento de estudio para la sustentación y referencia para el PPT.
> Cubre todo lo realizado desde la corrección de datasets hasta el notebook del Hito 2.

---

## 1. Correcciones a los Datasets (Pre-Hito 2)

Antes de avanzar al modelado se identificaron y corrigieron inconsistencias en **todos los datasets** para que sean coherentes con el contexto real de ALDIMI (albergue oncológico pediátrico).

---

### 1.1 `cancer-risk-factors.csv` — Nueva columna: `Metastasis_Status`

**Problema:** El dataset no tenía información sobre el estado de metástasis del paciente, una variable clínicamente relevante recomendada por el profesor.

**Solución:** Se añadió la columna `Metastasis_Status` (0 = sin metástasis, 1 = con metástasis) con **probabilidades condicionales** según el nivel de riesgo del paciente:

| Nivel de Riesgo | Probabilidad de Metástasis |
|---|---|
| High | 65% |
| Medium | 28% |
| Low | 8% |

**Resultado real en el dataset:**
| Risk_Level | Sin metástasis | Con metástasis | Tasa real |
|---|---|---|---|
| High | 18 | 84 | **82%** |
| Medium | 1,082 | 492 | **31%** |
| Low | 313 | 11 | **3%** |

**¿Por qué importa?** La correlación de `Metastasis_Status` con `Overall_Risk_Score` es **r = 0.34**, lo que confirma que el modelo puede aprender de esta variable. En la presentación: _"añadimos esta variable porque el estado de metástasis es un indicador clínico directo de la gravedad del paciente"_.

---

### 1.2 `inventory_data.csv` — Reemplazo de ítems no pertinentes

**Problema:** El ítem `X-ray Machine` no corresponde al contexto de ALDIMI. Un albergue no rastrea máquinas de rayos X como insumo consumible diario.

**Solución:** Las 98 filas de `X-ray Machine` fueron reemplazadas por 4 insumos propios de un albergue oncológico pediátrico:

| Nuevo ítem | Tipo | Costo unitario | Uso diario |
|---|---|---|---|
| Morfina | Consumable | S/. 15 – 85 | 5 – 40 unid/día |
| Sábanas | Consumable | S/. 18 – 55 | 3 – 20 unid/día |
| Alimento Complementario | Consumable | S/. 8 – 30 | 10 – 60 unid/día |
| Kit de Higiene | Consumable | S/. 5 – 28 | 8 – 45 unid/día |

**Distribución final del inventario:**

| Ítem | Registros |
|---|---|
| IV Drip | 109 |
| Gloves | 99 |
| Surgical Mask | 98 |
| Ventilator | 96 |
| Morfina | 25 |
| Sábanas | 25 |
| Alimento Complementario | 24 |
| Kit de Higiene | 24 |

---

### 1.3 `patient_data.csv` — Corrección integral

**Problema:** El dataset describía un hospital genérico, no un albergue oncológico. Tenía diagnósticos como _Diabetes_ y _Fractura_, procedimientos como _Appendectomy_, y **195 filas con fecha de alta anterior a la de ingreso**.

**Solución:**

| Campo | Antes | Después |
|---|---|---|
| `Primary_Diagnosis` | Diabetes, Fractura, Apendicitis, Neumonía | Leucemia Linfoblástica Aguda, Neuroblastoma, Osteosarcoma, Linfoma de Hodgkin, Tumor de Wilms, Meduloblastoma, Retinoblastoma, Leucemia Mieloide Aguda |
| `Procedure_Performed` | Appendectomy, Chest X-ray, MRI | Quimioterapia, Radioterapia, Transfusión de Sangre, Biopsia, Punción Lumbar, Análisis de Sangre, Control de Tratamiento |
| `Room_Type` | General Ward, ICU | Habitación Compartida, Habitación Individual, Sala de Tratamiento |
| `Equipment_Used` | Surgical Table, MRI Machine, X-ray Machine | Bomba de Infusión, Oxímetro, Monitor de Signos Vitales, Silla de Ruedas |
| `Staff_Needed` | 2 Surgeons, 1 Nurse | 1 Enfermero/a Oncológico, 1 Médico Oncólogo, 1 Nutricionista, 1 Trabajador/a Social |
| Fechas incoherentes | **195 filas** con Discharge < Admission | **0 filas** — reconstruidas con `Admission_Date + Bed_Days` |

---

### 1.4 `financial_data.csv` — Descripción de gastos

**Problema:** La descripción `"Surgeons' salaries"` no corresponde a ALDIMI.

**Cambio:**
| Descripción anterior | Descripción nueva |
|---|---|
| Surgical masks | Suministros médicos |
| Ventilators | Equipamiento médico |
| Surgeons' salaries | Sueldos del personal de apoyo |

---

### 1.5 `staff_data.csv` — Tipos de personal y asignaciones

**Problema:** El personal y las asignaciones describían un hospital de emergencias, no un albergue.

| Campo | Antes | Después |
|---|---|---|
| `Staff_Type` | Surgeon, Nurse, Technician | Médico Oncólogo, Enfermero/a Oncológico, Trabajador/a Social |
| `Current_Assignment` | ER, General Ward, ICU Surgery | Sala de Tratamiento, Área de Descanso, Sala Psicosocial |

---

### 1.6 `vendor_data.csv` — Proveedor V003

**Problema:** V003 suministraba `X-ray Machine` (ítem que eliminamos del inventario).

| | Antes | Después |
|---|---|---|
| Item_Supplied | X-ray Machine | Morfina |
| Cost_Per_Item | S/. 5,000 | S/. 45.50 |
| Avg_Lead_Time | 15 días | 3 días |

---

## 2. Actualizaciones al EDA del Hito 1

Se corrigieron **3 celdas** del notebook `Hito 1.ipynb` para mantener coherencia con los datasets actualizados:

### Celda 30 — Nueva observación sobre `Metastasis_Status`
Se añadió un tercer punto al análisis de correlaciones del frente de salud, documentando la nueva variable y su gradiente por nivel de riesgo (82% / 31% / 3%).

### Celda 38 — Distribución de ítems en inventario
Se actualizó para mencionar los nuevos insumos ALDIMI y la distribución real: **63% Consumibles** (antes la proporción era distinta con X-ray Machine).

### Celda 41 — Cifras de stock crítico corregidas
| Cifra | Antes | Después |
|---|---|---|
| Ítems en estado crítico | 43 (9%) | **58 (11.6%)** |
| Consumibles críticos | 27 | **45** |
| Equipos críticos | — | **13** |

---

## 3. Hito 2 — Notebook Completo

### Estructura del Notebook (`Hito 2.ipynb`, 37 celdas)

```
Setup e imports
├── Fase 3: Preparación de Datos
│   ├── 3.1 Frente de Salud
│   │   ├── Revisión de nulos (resultado: 0 nulos)
│   │   ├── Encoding de Cancer_Type
│   │   ├── Feature Engineering (3 variables nuevas)
│   │   ├── Análisis de desbalanceo de clases
│   │   └── Train / Test Split (80/20, estratificado)
│   └── 3.2 Frente Logístico
│       ├── Parsing de fechas + features temporales
│       ├── Feature Engineering (6 variables nuevas)
│       ├── Encoding de Item_Name e Item_Type
│       └── Train / Test Split (80/20)
└── Fase 4: Modelado Baseline
    ├── 4.1 Árbol de Decisión — Clasificación de Riesgo
    │   ├── Métricas: Accuracy, F1-Score, Classification Report
    │   ├── Matriz de confusión
    │   └── Importancia de variables
    ├── 4.2 Regresión Lineal — Predicción de Stock
    │   ├── Métricas: MAE, RMSE, R² (7 días y 14 días)
    │   └── Gráfico Real vs Predicho
    └── Dashboard Preliminar (4 paneles)
```

---

## 4. Fase 3: Preparación de Datos — Detalle

### 4.1 Frente de Salud

#### Encoding
`Cancer_Type` (única variable categórica no numérica) → **Label Encoding**:

| Tipo de Cáncer | Código |
|---|---|
| Breast | 0 |
| Cervical | 1 |
| Colorectal | 2 |
| Liver | 3 |
| Lung | 4 |
| Prostate | 5 |
| Skin | 6 |
| Stomach | 7 |

#### Feature Engineering — Variables Compuestas

Se crearon **3 variables nuevas** que condensan patrones clínicos:

| Variable | Fórmula | Qué captura |
|---|---|---|
| `Risk_Lifestyle` | `Smoking + Alcohol_Use + Air_Pollution` | Carga acumulada de riesgo ambiental y conductual |
| `Diet_Score` | `Diet_Salted_Processed + Diet_Red_Meat - Fruit_Veg_Intake` | Calidad dietética neta (positivo = dieta dañina) |
| `Genetic_Risk` | `BRCA_Mutation + Family_History + H_Pylori_Infection` | Predisposición genética y biológica acumulada |

**¿Por qué esto ayuda al modelo?** En lugar de que el árbol evalúe 3 variables separadas, evalúa una sola que ya condensa su interacción — reduce la profundidad necesaria del árbol y mejora la generalización.

#### Desbalanceo de clases
| Clase | N | % |
|---|---|---|
| Medium | ~1,574 | ~79% |
| Low | ~324 | ~16% |
| High | ~102 | ~5% |

**Estrategia:** `class_weight='balanced'` en el DecisionTreeClassifier. Esto hace que el modelo penalice más los errores en las clases minoritarias (High, Low), que son las más críticas para ALDIMI.

#### Split Train/Test
- **80% Train / 20% Test**
- `stratify=y` → mantiene la proporción de clases en ambos subsets
- `random_state=42` → reproducibilidad

---

### 4.2 Frente Logístico

#### Features Temporales (desde la columna `Date`)
| Feature | Descripción |
|---|---|
| `Month` | Mes del registro (1–12) — captura estacionalidad |
| `Quarter` | Trimestre (1–4) |
| `Day_of_Week` | Día de la semana (0=Lunes, 6=Domingo) |

#### Feature Engineering — Variables de Inventario

| Variable | Fórmula | Propósito |
|---|---|---|
| `Projected_Stock_7d` | `max(0, Current_Stock - Avg_Usage_Per_Day × 7)` | Stock estimado en 7 días |
| `Projected_Stock_14d` | `max(0, Current_Stock - Avg_Usage_Per_Day × 14)` | Stock estimado en 14 días |
| `Days_Until_Stockout` | `Current_Stock / Avg_Usage_Per_Day` | Días hasta agotamiento |
| `Stock_Ratio` | `Current_Stock / Min_Required` | Ratio de seguridad (< 1 = crítico) |
| `Is_Critical_7d` | `1 si Projected_Stock_7d < Min_Required` | Flag de alerta a 7 días |
| `Is_Critical_14d` | `1 si Projected_Stock_14d < Min_Required` | Flag de alerta a 14 días |

#### Alertas de stock detectadas
| Horizonte | Ítems en alerta | % del inventario |
|---|---|---|
| 7 días | **259** | **51.8%** |
| 14 días | **382** | **76.4%** |

> **Punto de sustentación:** Más de la mitad del inventario estará en estado crítico en menos de una semana. Esto demuestra la urgencia del sistema predictivo para ALDIMI.

---

## 5. Fase 4: Modelos Baseline — Resultados

### 5.1 Clasificación de Riesgo de Salud
**Modelo:** `DecisionTreeClassifier(max_depth=5, class_weight='balanced')`

| Métrica | Valor |
|---|---|
| **Accuracy** | **99.75%** |
| **F1-Score macro** | **99.69%** |
| F1 — Low | 0.99 |
| F1 — Medium | 1.00 |
| F1 — High | 1.00 |

**Variable más importante:** `Overall_Risk_Score` (score continuo que agrega todos los factores de riesgo).

> **Nota para la sustentación:** El accuracy muy alto se debe a que `Overall_Risk_Score` es la variable continua de la que se deriva `Risk_Level`. El modelo aprende los umbrales de corte casi perfectamente. En el **Hito 3** se evaluará el modelo sin esta variable para medir la robustez real con variables clínicas puras.

---

### 5.2 Predicción de Stock Logístico
**Modelo:** `LinearRegression()` — dos instancias (una por horizonte)

| Horizonte | MAE | RMSE | R² |
|---|---|---|---|
| **7 días** | 364 unidades | 471 unidades | **0.82** |
| **14 días** | 387 unidades | 499 unidades | **0.64** |

> **Interpretación:** El R² baja de 0.82 a 0.64 al aumentar el horizonte de 7 a 14 días — esperable porque hay más incertidumbre a mayor plazo. En el **Hito 3** se usarán modelos de series temporales (ARIMA / Prophet) para capturar patrones estacionales de demanda real.

---

## 6. Dashboard Preliminar

El dashboard tiene **4 paneles** y se guarda como `dashboard_hito2.png`:

| Panel | Contenido | Propósito |
|---|---|---|
| Superior izq. | Riesgo de Salud: Real vs Predicho (barras agrupadas) | Validar que el modelo predice correctamente las 3 clases |
| Superior der. | Alertas de Stock Crítico por ítem (7d y 14d) | Identificar qué insumos necesitan reabastecimiento urgente |
| Inferior izq. | Top 8 Variables Predictivas — Riesgo de Salud | Explicar qué factores determinan el nivel de riesgo |
| Inferior der. | Stock Promedio: Actual vs +7d vs +14d por ítem | Visualizar la caída proyectada de inventario por insumo |

---

## 7. Próximos Pasos — Hito 3

| Frente | Hito 2 (Baseline) | Hito 3 (Avanzado) |
|---|---|---|
| Clasificación | Árbol de Decisión | **XGBoost / Random Forest** + SMOTE + GridSearchCV |
| Logística | Regresión Lineal | **ARIMA / Prophet** para series temporales reales |
| Evaluación | Métricas básicas | Validación cruzada + análisis de errores |
| Integración | — | Conexión con DB compartida del equipo de IA |

---

## 8. Puntos Clave para la Sustentación (Hito 2)

### Pitch de impacto (primeros 2 minutos)
- ALDIMI atiende a **niños con cáncer en extrema pobreza**. Un desabastecimiento de Morfina o Alimento Complementario tiene consecuencias directas en su bienestar.
- Nuestro sistema predice el stock a **7 y 14 días** y clasifica el nivel de riesgo clínico de cada paciente.
- **ODS 3 (Salud):** El modelo de clasificación permite priorizar la atención de los pacientes más críticos.
- **ODS 10 (Reducción de Desigualdades):** El modelo logístico optimiza los recursos limitados del albergue.

### Puntos técnicos que defender
1. **¿Por qué Árbol de Decisión como baseline?** Es interpretable — los directivos de ALDIMI pueden entender sus reglas de decisión sin conocer ML.
2. **¿Por qué class_weight='balanced'?** El 5% de pacientes High son los más críticos. Sin este ajuste, el modelo los ignoraría priorizando la clase mayoritaria.
3. **¿Por qué crearon variables compuestas?** `Risk_Lifestyle`, `Diet_Score` y `Genetic_Risk` consolidan factores correlacionados, reducen ruido y mejoran la capacidad del árbol para generalizar.
4. **¿Por qué el R² baja de 7 a 14 días?** Mayor horizonte = mayor incertidumbre acumulada. La regresión lineal no captura variabilidad no-lineal — por eso en Hito 3 usaremos series temporales.
5. **¿Por qué el 76% de ítems en alerta a 14 días?** El promedio de `Days_Until_Stockout` es ~20 días, y el percentil 25 es solo 4.4 días. Gran parte del inventario ya opera cerca del límite mínimo.

---

## 9. Resumen de Archivos Modificados / Creados

| Archivo | Acción | Descripción |
|---|---|---|
| `cancer-risk-factors.csv` | Modificado | +1 columna: `Metastasis_Status` |
| `inventory_data.csv` | Modificado | 98 filas de X-ray Machine → insumos ALDIMI |
| `patient_data.csv` | Modificado | Diagnósticos, procedimientos, fechas, personal corregidos |
| `financial_data.csv` | Modificado | Descripción de gastos actualizada |
| `staff_data.csv` | Modificado | Tipos de personal y asignaciones actualizadas |
| `vendor_data.csv` | Modificado | V003: X-ray Machine → Morfina |
| `Hito 1.ipynb` | Modificado | Celdas 30, 38, 41 actualizadas con nuevas cifras |
| `Hito 2.ipynb` | Creado | Notebook completo Fases 3 y 4 de CRISP-DM (37 celdas) |
| `dashboard_hito2.png` | Creado | Dashboard de 4 paneles generado automáticamente |
