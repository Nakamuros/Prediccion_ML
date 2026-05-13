import json

with open('Hito 1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the conclusion cells
conc_salud = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Conclusiones del Análisis Bivariado:**\n",
        "\n",
        "* **El Cáncer de Pulmón concentra la mayor criticidad:** Casi la mitad de los pacientes clasificados con Riesgo Alto padecen cáncer de pulmón. El modelo deberá dar especial peso a este diagnóstico para prever recursos críticos.\n",
        "* **La Edad no es un predictor fuerte del nivel de riesgo:** La edad media (63.7 años) y la mediana (64 años) son virtualmente idénticas en pacientes de todos los niveles de riesgo. Otros factores clínicos o de estilo de vida traccionan el riesgo, no la edad."
    ]
}

conc_inv = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Conclusiones del Análisis Logístico:**\n",
        "\n",
        "* **Tiempos de Reabastecimiento uniformes:** Los proveedores tardan en promedio 15 días en despachar tanto Equipos como Consumibles. Las alertas de predicción deberán emitirse con al menos 2 a 3 semanas de anticipación para evitar desabastecimientos.\n",
        "* **Alerta de Stock Crítico:** Existen 43 artículos (casi el 9% del inventario) con un Ratio de Stock <= 1 (stock menor o igual al mínimo). De ellos, la mayoría (27) son consumibles. El modelo predictivo debe enfocarse fuertemente en esta categoría, pues representan suministros vitales (medicamentos, alimentos) y están al borde del desabastecimiento."
    ]
}

idx_salud = -1
idx_inv = -1

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = "".join(c['source'])
        if "sns.countplot(data=df_cancer, x='Cancer_Type', hue='Risk_Level'" in source:
            idx_salud = i
        if "sns.boxplot(data=df_inventory, x='Item_Type', y='Restock_Lead_Time'" in source:
            idx_inv = i

# We must insert backwards so indices don't shift for the first insertion
if idx_inv != -1:
    nb['cells'].insert(idx_inv + 1, conc_inv)

if idx_salud != -1:
    nb['cells'].insert(idx_salud + 1, conc_salud)

with open('Hito 1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
