import json

with open('Hito 1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

idx_cancer = -1
idx_outliers = -1

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown' and '# **CANCER**' in "".join(c['source']):
        idx_cancer = i
    if c['cell_type'] == 'code' and 'fig, axes = plt.subplots(2, 1,' in "".join(c['source']):
        idx_outliers = i

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code + "\n"]}

stats_cells = [
    md_cell("# **1. ESTADÍSTICAS INICIALES: RIESGO DE CÁNCER**"),
    md_cell("### 1.1 Vista Previa de los Datos\nInspección de las primeras filas del dataset para comprender su estructura y las variables disponibles."),
    code_cell("df_cancer.head()"),
    
    md_cell("### 1.2 Información del Dataset y Tipos de Datos\nRevisión de los tipos de datos y la memoria utilizada, útil para identificar si hay variables categóricas mal formateadas."),
    code_cell("df_cancer.info()"),
    
    md_cell("### 1.3 Estadísticas Descriptivas\nResumen estadístico de las variables numéricas (media, desviación estándar, cuartiles)."),
    code_cell("df_cancer.describe()"),
    
    md_cell("### 1.4 Valores Nulos por Columna\nVerificación de la existencia de datos faltantes en el dataset de salud."),
    code_cell("df_cancer.isna().sum()"),
    
    md_cell("### 1.5 Porcentaje de Valores en Cero\nAnálisis de la proporción de ceros en cada columna, lo que puede indicar categorías ausentes, valores nulos implícitos o características esparcidas."),
    code_cell("(df_cancer == 0).mean() * 100"),
    
    md_cell("# **2. ESTADÍSTICAS INICIALES: INVENTARIO LOGÍSTICO**"),
    md_cell("### 2.1 Vista Previa de los Datos\nInspección de las primeras filas del dataset logístico."),
    code_cell("df_inventory.head()"),
    
    md_cell("### 2.2 Información del Dataset y Tipos de Datos\nRevisión general de la estructura y tipos de variables del inventario."),
    code_cell("df_inventory.info()"),
    
    md_cell("### 2.3 Estadísticas Descriptivas\nResumen estadístico de las cantidades de inventario, costos y uso."),
    code_cell("df_inventory.describe()"),
    
    md_cell("### 2.4 Valores Nulos por Columna\nVerificación de datos faltantes en el control de inventario."),
    code_cell("df_inventory.isna().sum()"),
    
    md_cell("### 2.5 Porcentaje de Valores en Cero\nIdentificación de posibles faltas de stock (ceros) o datos de uso incompletos."),
    code_cell("(df_inventory == 0).mean() * 100")
]

if idx_cancer != -1 and idx_outliers != -1:
    # Replace the old messy stats cells with the new structured ones
    nb['cells'] = nb['cells'][:idx_cancer] + stats_cells + nb['cells'][idx_outliers:]

bivariate_salud = [
    md_cell("### Análisis Bivariado: Nivel de Riesgo vs Tipo de Cáncer y Edad\nExploración de cómo se relaciona el nivel de prioridad o riesgo con el tipo de cáncer y la edad del paciente."),
    code_cell("fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\nsns.countplot(data=df_cancer, x='Cancer_Type', hue='Risk_Level', ax=axes[0], palette='Reds')\naxes[0].set_title('Nivel de Riesgo según Tipo de Cáncer')\n\nsns.boxplot(data=df_cancer, x='Risk_Level', y='Age', ax=axes[1], palette='Blues')\naxes[1].set_title('Distribución de Edad según Nivel de Riesgo')\n\nplt.tight_layout()\nplt.show()")
]

bivariate_inv = [
    md_cell("### Análisis de Reabastecimiento y Stock por Tipo de Artículo\nEvaluación del tiempo de reabastecimiento y la relación de stock crítico según la categoría del artículo."),
    code_cell("fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\nsns.boxplot(data=df_inventory, x='Item_Type', y='Restock_Lead_Time', ax=axes[0], palette='Set2')\naxes[0].set_title('Tiempo de Reabastecimiento por Tipo de Artículo')\n\n# Crear variable temporal de estado de stock (Stock actual / Mínimo requerido)\ndf_inventory['Stock_Ratio'] = df_inventory['Current_Stock'] / df_inventory['Min_Required']\nsns.histplot(data=df_inventory, x='Stock_Ratio', hue='Item_Type', kde=True, ax=axes[1], palette='Set1')\naxes[1].set_title('Distribución del Ratio de Stock (Actual / Mínimo)')\naxes[1].axvline(1, color='red', linestyle='--', label='Límite Crítico (Ratio=1)')\naxes[1].legend()\n\nplt.tight_layout()\nplt.show()")
]

idx_eda_logistico = -1
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown' and '# **EDA FRENTE LOGÍSTICO**' in "".join(c['source']):
        idx_eda_logistico = i
        break

if idx_eda_logistico != -1:
    nb['cells'] = nb['cells'][:idx_eda_logistico] + bivariate_salud + nb['cells'][idx_eda_logistico:]

nb['cells'].extend(bivariate_inv)

with open('Hito 1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
