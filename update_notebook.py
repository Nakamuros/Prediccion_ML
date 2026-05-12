import json

with open('Hito 1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Add Objectives at the beginning
obj_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# **Definición de Objetivos de ML (Alineados con ODS 2 y 3)**\n",
        "\n",
        "El proyecto ALDIMI Predict busca construir un ecosistema de gestión inteligente mediante Inteligencia Artificial y Machine Learning. Los objetivos de Machine Learning del proyecto son:\n",
        "\n",
        "1. **Predicción de Demanda y Logística (Alineado con ODS 2 - Hambre Cero):** Desarrollar un modelo de predicción (regresión o series temporales) para anticipar la demanda de inventario (medicamentos y alimentos) a 7 y 14 días. Esto optimiza el abastecimiento y reduce el riesgo de desabastecimiento, asegurando nutrición y acceso a insumos básicos para las familias.\n",
        "\n",
        "2. **Clasificación de Riesgo de Salud (Alineado con ODS 3 - Salud y Bienestar):** Entrenar un modelo de clasificación que permita identificar de forma temprana el 'Nivel de Prioridad de Atención' (Bajo, Medio, Alto) de los pacientes según sus variables clínicas y sociales. Esto facilita una respuesta médica oportuna y eficiente para evitar complicaciones clínicas."
    ]
}
nb['cells'].insert(0, obj_cell)

# Fix the percentages and typos
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "(df_cancer == 0).sum()/100" in source:
            cell['source'] = ["(df_cancer == 0).mean() * 100"]
            cell['outputs'] = []
            cell['execution_count'] = None
        elif "(df_inventory == 0).sum()/100" in source:
            cell['source'] = ["(df_inventory == 0).mean() * 100"]
            cell['outputs'] = []
            cell['execution_count'] = None
    elif cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if '# **EDA FRENTE LOGSITCICO**' in source:
            cell['source'] = ['# **EDA FRENTE LOGÍSTICO**']

dist_cells_cancer = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Distribuciones Iniciales de Factores de Riesgo\n",
            "Análisis de la distribución de las variables numéricas más relevantes en el dataset de salud."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "sns.histplot(df_cancer['Age'], bins=20, kde=True, ax=axes[0], color='skyblue')\n",
            "axes[0].set_title('Distribución de Edad')\n",
            "sns.histplot(df_cancer['BMI'], bins=20, kde=True, ax=axes[1], color='lightgreen')\n",
            "axes[1].set_title('Distribución de BMI (Índice de Masa Corporal)')\n",
            "sns.histplot(df_cancer['Overall_Risk_Score'], bins=20, kde=True, ax=axes[2], color='salmon')\n",
            "axes[2].set_title('Distribución del Score de Riesgo General')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

dist_cells_inventory = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Distribuciones Iniciales del Inventario\n",
            "Análisis de la distribución del inventario y el uso promedio para evaluar la demanda."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
            "sns.histplot(df_inventory['Current_Stock'], bins=20, kde=True, ax=axes[0], color='blue')\n",
            "axes[0].set_title('Distribución de Stock Actual')\n",
            "sns.histplot(df_inventory['Avg_Usage_Per_Day'], bins=20, kde=True, ax=axes[1], color='orange')\n",
            "axes[1].set_title('Distribución de Uso Promedio por Día')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

# Insert distributions
idx_salud = -1
idx_inv = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if '# **EDA RIESGO DE SALUD**' in source:
            idx_salud = i
        elif '# **EDA FRENTE LOGÍSTICO**' in source:
            idx_inv = i

if idx_inv != -1:
    nb['cells'] = nb['cells'][:idx_inv+1] + dist_cells_inventory + nb['cells'][idx_inv+1:]
if idx_salud != -1:
    nb['cells'] = nb['cells'][:idx_salud+1] + dist_cells_cancer + nb['cells'][idx_salud+1:]

with open('Hito 1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
