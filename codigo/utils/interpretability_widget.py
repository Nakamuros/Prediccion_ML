"""Widget reutilizable de importancia de features, usado en las páginas
de Alertas de Stock y Riesgo de Pacientes para dar contexto de qué
variables explican la predicción del modelo correspondiente."""
import pandas as pd
import plotly.express as px
from utils.labels import pretty_feature_name


def render_feature_importance(st_container, importances: dict, top_n: int = 8):
    """Dibuja un gráfico de barras horizontal + tabla de ranking con las
    features más importantes de un modelo, traducidas a español.

    st_container: puede ser `st` directamente o un `st.container()`/columna.
    importances: dict {nombre_feature: valor_importancia}
    """
    if not importances:
        st_container.info("El modelo ganador no expone feature_importances_.")
        return

    df_imp = (
        pd.Series(importances)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    df_imp.columns = ['Feature', 'Importancia']
    df_imp['Feature'] = df_imp['Feature'].apply(pretty_feature_name)

    df_plot = df_imp.sort_values('Importancia', ascending=True)
    fig = px.bar(df_plot, x='Importancia', y='Feature', orientation='h')
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="",
        xaxis_title="Importancia relativa",
        height=320,
    )
    fig.update_traces(marker_color="#3D9DF3")
    st_container.plotly_chart(fig, width='stretch')