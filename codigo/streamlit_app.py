from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:  # pragma: no cover - optional dependency
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SMOTE = False


APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT.parent / "datos"

st.set_page_config(
    page_title="ALDIMI Predict | Hito 3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cancer = pd.read_csv(DATA_ROOT / "cancer-risk-factors.csv")
    inventory = pd.read_csv(DATA_ROOT / "inventory_data.csv")
    return cancer, inventory


def build_health_features(df: pd.DataFrame):
    df = df.copy()
    df["Risk_Lifestyle"] = df["Smoking"] + df["Alcohol_Use"] + df["Air_Pollution"]
    df["Diet_Score"] = df["Diet_Salted_Processed"] + df["Diet_Red_Meat"] - df["Fruit_Veg_Intake"]
    df["Genetic_Risk"] = df["BRCA_Mutation"] + df["Family_History"] + df["H_Pylori_Infection"]

    le_cancer_type = LabelEncoder()
    df["Cancer_Type_enc"] = le_cancer_type.fit_transform(df["Cancer_Type"])

    feature_cols = [
        "Age",
        "Gender",
        "Smoking",
        "Alcohol_Use",
        "Obesity",
        "Family_History",
        "Diet_Red_Meat",
        "Diet_Salted_Processed",
        "Fruit_Veg_Intake",
        "Physical_Activity",
        "Air_Pollution",
        "Occupational_Hazards",
        "BRCA_Mutation",
        "H_Pylori_Infection",
        "Calcium_Intake",
        "BMI",
        "Physical_Activity_Level",
        "Cancer_Type_enc",
        "Risk_Lifestyle",
        "Diet_Score",
        "Genetic_Risk",
    ]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Risk_Level"])
    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dt_base = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    dt_base.fit(X_train, y_train)
    baseline_f1 = f1_score(y_test, dt_base.predict(X_test), average="macro")

    return {
        "df": df,
        "encoder": target_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "dt_base": dt_base,
        "baseline_f1": baseline_f1,
    }


def build_inventory_features(df: pd.DataFrame):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Projected_Stock_7d"] = (df["Current_Stock"] - df["Avg_Usage_Per_Day"] * 7).clip(lower=0)
    df["Projected_Stock_14d"] = (df["Current_Stock"] - df["Avg_Usage_Per_Day"] * 14).clip(lower=0)
    df["Days_Until_Stockout"] = (df["Current_Stock"] / df["Avg_Usage_Per_Day"]).round(2)
    df["Stock_Ratio"] = (df["Current_Stock"] / df["Min_Required"]).round(3)

    item_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    df["Item_Name_enc"] = item_encoder.fit_transform(df["Item_Name"])
    df["Item_Type_enc"] = type_encoder.fit_transform(df["Item_Type"])

    feature_cols = [
        "Current_Stock",
        "Min_Required",
        "Max_Capacity",
        "Unit_Cost",
        "Avg_Usage_Per_Day",
        "Restock_Lead_Time",
        "Item_Name_enc",
        "Item_Type_enc",
        "Month",
        "Quarter",
        "Day_of_Week",
        "Stock_Ratio",
        "Days_Until_Stockout",
    ]

    X = df[feature_cols]
    y7 = df["Projected_Stock_7d"]
    y14 = df["Projected_Stock_14d"]
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train_7 = y7.iloc[train_idx]
    y_test_7 = y7.iloc[test_idx]
    y_train_14 = y14.iloc[train_idx]
    y_test_14 = y14.iloc[test_idx]

    lr_7 = LinearRegression().fit(X_train, y_train_7)
    lr_14 = LinearRegression().fit(X_train, y_train_14)
    baseline_mae_7 = mean_absolute_error(y_test_7, lr_7.predict(X_test))
    baseline_mae_14 = mean_absolute_error(y_test_14, lr_14.predict(X_test))
    baseline_r2_7 = r2_score(y_test_7, lr_7.predict(X_test))
    baseline_r2_14 = r2_score(y_test_14, lr_14.predict(X_test))

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_7": y_train_7,
        "y_test_7": y_test_7,
        "y_train_14": y_train_14,
        "y_test_14": y_test_14,
        "feature_cols": feature_cols,
        "lr_7": lr_7,
        "lr_14": lr_14,
        "baseline_mae_7": baseline_mae_7,
        "baseline_mae_14": baseline_mae_14,
        "baseline_r2_7": baseline_r2_7,
        "baseline_r2_14": baseline_r2_14,
    }


@st.cache_resource(show_spinner=False)
def train_health_models(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, baseline_f1: float):
    if HAS_SMOTE:
        X_train_sm, y_train_sm = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train.copy(), y_train.copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced"],
        },
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    rf_search.fit(X_train_sm, y_train_sm)
    best_rf = rf_search.best_estimator_
    pred_rf = best_rf.predict(X_test)
    rf_f1 = f1_score(y_test, pred_rf, average="macro")

    if HAS_XGB:
        xgb_base = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42, verbosity=0)
        xgb_space = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
        }
    else:
        xgb_base = XGBClassifier(random_state=42)
        xgb_space = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
        }

    xgb_search = RandomizedSearchCV(
        xgb_base,
        xgb_space,
        n_iter=12,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
    )
    xgb_search.fit(X_train_sm, y_train_sm)
    best_xgb = xgb_search.best_estimator_
    pred_xgb = best_xgb.predict(X_test)
    xgb_f1 = f1_score(y_test, pred_xgb, average="macro")

    return {
        "best_rf": best_rf,
        "pred_rf": pred_rf,
        "rf_f1": rf_f1,
        "best_xgb": best_xgb,
        "pred_xgb": pred_xgb,
        "xgb_f1": xgb_f1,
        "baseline_f1": baseline_f1,
    }


@st.cache_resource(show_spinner=False)
def train_inventory_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train_7: pd.Series, y_test_7: pd.Series, y_train_14: pd.Series, y_test_14: pd.Series):
    cv = 5

    rf_search_7 = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]},
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rf_search_7.fit(X_train, y_train_7)
    best_rf_7 = rf_search_7.best_estimator_
    pred_rf_7 = best_rf_7.predict(X_test)

    rf_search_14 = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]},
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rf_search_14.fit(X_train, y_train_14)
    best_rf_14 = rf_search_14.best_estimator_
    pred_rf_14 = best_rf_14.predict(X_test)

    if HAS_XGB:
        xgb_base = XGBRegressor(random_state=42, verbosity=0)
        xgb_space = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
    else:
        xgb_base = XGBRegressor(random_state=42)
        xgb_space = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}

    xgb_search_7 = RandomizedSearchCV(
        xgb_base,
        xgb_space,
        n_iter=10,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )
    xgb_search_7.fit(X_train, y_train_7)
    best_xgb_7 = xgb_search_7.best_estimator_
    pred_xgb_7 = best_xgb_7.predict(X_test)

    xgb_search_14 = RandomizedSearchCV(
        xgb_base,
        xgb_space,
        n_iter=10,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )
    xgb_search_14.fit(X_train, y_train_14)
    best_xgb_14 = xgb_search_14.best_estimator_
    pred_xgb_14 = best_xgb_14.predict(X_test)

    return {
        "lr_7": LinearRegression().fit(X_train, y_train_7),
        "lr_14": LinearRegression().fit(X_train, y_train_14),
        "pred_rf_7": pred_rf_7,
        "pred_rf_14": pred_rf_14,
        "pred_xgb_7": pred_xgb_7,
        "pred_xgb_14": pred_xgb_14,
        "rf_mae_7": mean_absolute_error(y_test_7, pred_rf_7),
        "rf_mae_14": mean_absolute_error(y_test_14, pred_rf_14),
        "rf_r2_7": r2_score(y_test_7, pred_rf_7),
        "rf_r2_14": r2_score(y_test_14, pred_rf_14),
        "xgb_mae_7": mean_absolute_error(y_test_7, pred_xgb_7),
        "xgb_mae_14": mean_absolute_error(y_test_14, pred_xgb_14),
        "xgb_r2_7": r2_score(y_test_7, pred_xgb_7),
        "xgb_r2_14": r2_score(y_test_14, pred_xgb_14),
    }


def plot_confusion(ax, y_true, y_pred, labels, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)


def health_section(df_cancer: pd.DataFrame, encoder: LabelEncoder, X_test: pd.DataFrame, y_test: np.ndarray, health_models: dict):
    st.subheader("Frente de Salud")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline F1-macro", f"{health_models['baseline_f1']:.3f}")
    c2.metric("Random Forest", f"{health_models['rf_f1']:.3f}", delta=f"{health_models['rf_f1'] - health_models['baseline_f1']:+.3f}")
    c3.metric("XGBoost", f"{health_models['xgb_f1']:.3f}", delta=f"{health_models['xgb_f1'] - health_models['baseline_f1']:+.3f}")
    c4.metric("Registros", f"{len(df_cancer):,}")

    tab1, tab2, tab3 = st.tabs(["Comparativa", "Errores", "Importancia"])

    with tab1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        model_items = [
            ("Árbol Dec. (Baseline)", health_models["dt_base"].predict(X_test)),
            ("Random Forest", health_models["pred_rf"]),
            ("XGBoost", health_models["pred_xgb"]),
        ]
        for ax, (name, pred) in zip(axes, model_items):
            plot_confusion(ax, y_test, pred, encoder.classes_, name)
            f1m = f1_score(y_test, pred, average="macro")
            idx_high = list(encoder.classes_).index("High")
            f1h = f1_score(y_test, pred, average=None)[idx_high]
            ax.set_title(f"{name}\nF1-macro={f1m:.3f} · F1-High={f1h:.3f}")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        summary = pd.DataFrame(
            {
                "Modelo": ["Árbol Dec.", "Random Forest", "XGBoost"],
                "F1-macro": [
                    f1_score(y_test, health_models["dt_base"].predict(X_test), average="macro"),
                    health_models["rf_f1"],
                    health_models["xgb_f1"],
                ],
                "Accuracy": [
                    accuracy_score(y_test, health_models["dt_base"].predict(X_test)),
                    accuracy_score(y_test, health_models["pred_rf"]),
                    accuracy_score(y_test, health_models["pred_xgb"]),
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with tab2:
        best_pred = health_models["pred_xgb"] if health_models["xgb_f1"] >= health_models["rf_f1"] else health_models["pred_rf"]
        err_df = X_test.copy().reset_index(drop=True)
        err_df["Real"] = encoder.inverse_transform(y_test)
        err_df["Predicho"] = encoder.inverse_transform(best_pred)
        err_df["Error"] = err_df["Real"] != err_df["Predicho"]
        st.write(f"Errores totales: **{int(err_df['Error'].sum())}** / **{len(err_df)}**")

        error_types = err_df[err_df["Error"]].groupby(["Real", "Predicho"]).size().sort_values(ascending=False)
        if len(error_types) > 0:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = [f"{r}→{p}" for r, p in error_types.index]
            ax.barh(labels, error_types.values, color=["crimson" if "High" in label else "steelblue" for label in labels])
            ax.set_title("Errores por tipo")
            ax.set_xlabel("Conteo")
            st.pyplot(fig, clear_figure=True)
        st.dataframe(err_df.head(20), use_container_width=True)

    with tab3:
        if hasattr(health_models["best_rf"], "feature_importances_"):
            imp = pd.Series(health_models["best_rf"].feature_importances_, index=health_models["feature_cols"]).sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(8.5, 5))
            imp.plot(kind="barh", ax=ax, color=["mediumseagreen" if v > imp.mean() else "steelblue" for v in imp.values])
            ax.set_title("Importancia de variables — Random Forest")
            ax.set_xlabel("Importancia relativa")
            st.pyplot(fig, clear_figure=True)


def inventory_section(df_inv: pd.DataFrame, X_test: pd.DataFrame, y_test_7: pd.Series, y_test_14: pd.Series, inventory_models: dict):
    st.subheader("Frente Logístico")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline MAE 7d", f"{inventory_models['baseline_mae_7']:.1f}")
    c2.metric("Random Forest 7d", f"{inventory_models['rf_mae_7']:.1f}", delta=f"{inventory_models['baseline_mae_7'] - inventory_models['rf_mae_7']:+.1f}")
    c3.metric("XGBoost 7d", f"{inventory_models['xgb_mae_7']:.1f}", delta=f"{inventory_models['baseline_mae_7'] - inventory_models['xgb_mae_7']:+.1f}")
    c4.metric("Baseline MAE 14d", f"{inventory_models['baseline_mae_14']:.1f}")

    tab1, tab2, tab3 = st.tabs(["Comparativa", "Serie temporal", "Alertas"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        preds = {
            "Reg. Lineal": (inventory_models["lr_7"].predict(X_test), inventory_models["lr_14"].predict(X_test)),
            "Random Forest": (inventory_models["pred_rf_7"], inventory_models["pred_rf_14"]),
            "XGBoost": (inventory_models["pred_xgb_7"], inventory_models["pred_xgb_14"]),
        }
        for ax_idx, (horizon, y_true) in enumerate([("7 días", y_test_7), ("14 días", y_test_14)]):
            ax = axes[ax_idx]
            maes = [mean_absolute_error(y_true, preds[name][ax_idx]) for name in preds]
            names = list(preds.keys())
            bars = ax.bar(names, maes, color=["#6baed6", "#fd8d3c", "#74c476"], edgecolor="white", width=0.6)
            for bar, mae in zip(bars, maes):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(maes) * 0.02, f"{mae:.0f}", ha="center", va="bottom")
            ax.set_title(f"MAE a {horizon}")
            ax.set_ylabel("MAE")
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        summary = pd.DataFrame(
            {
                "Modelo": ["Reg. Lineal", "Random Forest", "XGBoost"],
                "MAE 7d": [
                    mean_absolute_error(y_test_7, inventory_models["lr_7"].predict(X_test)),
                    inventory_models["rf_mae_7"],
                    inventory_models["xgb_mae_7"],
                ],
                "R² 7d": [
                    r2_score(y_test_7, inventory_models["lr_7"].predict(X_test)),
                    inventory_models["rf_r2_7"],
                    inventory_models["xgb_r2_7"],
                ],
                "MAE 14d": [
                    mean_absolute_error(y_test_14, inventory_models["lr_14"].predict(X_test)),
                    inventory_models["rf_mae_14"],
                    inventory_models["xgb_mae_14"],
                ],
                "R² 14d": [
                    r2_score(y_test_14, inventory_models["lr_14"].predict(X_test)),
                    inventory_models["rf_r2_14"],
                    inventory_models["xgb_r2_14"],
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with tab2:
        items = sorted(df_inv["Item_Name"].unique())
        selected_items = st.multiselect("Selecciona insumos", items, default=items[:4])
        fig, ax = plt.subplots(figsize=(12, 5))
        for item in selected_items:
            df_item = df_inv[df_inv["Item_Name"] == item].sort_values("Date")
            ax.plot(df_item["Date"], df_item["Current_Stock"], label=item)
        ax.set_title("Evolución temporal del stock")
        ax.set_ylabel("Current_Stock")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with tab3:
        df_eval = df_inv.sort_values("Date").reset_index(drop=True)
        cut = int(len(df_eval) * 0.8)
        df_test_eval = df_eval.iloc[cut:].copy()
        df_test_eval["Pred_7d"] = inventory_models["pred_rf_7"]
        df_test_eval["Alert_Pred_7d"] = (df_test_eval["Pred_7d"] < df_test_eval["Min_Required"]).astype(int)
        alerts = df_test_eval.groupby("Item_Name")["Alert_Pred_7d"].sum().sort_values(ascending=False)
        st.dataframe(alerts.reset_index(name="Días críticos"), use_container_width=True, hide_index=True)
        if len(alerts) > 0:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            alerts.plot(kind="bar", ax=ax, color="tomato", edgecolor="white")
            ax.set_ylabel("Días críticos")
            ax.set_title("Alertas de stock crítico a 7 días")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)


def main():
    st.title("ALDIMI Predict | Hito 3")
    st.caption("Primera versión del dashboard en Streamlit para salud y logística.")

    df_cancer, df_inv = load_data()
    health = build_health_features(df_cancer)
    inventory = build_inventory_features(df_inv)

    with st.sidebar:
        st.header("Navegación")
        page = st.radio("Sección", ["Resumen", "Salud", "Logística", "Datos"], index=0)
        st.divider()
        st.markdown("### Estado")
        st.write(f"XGBoost disponible: {'Sí' if HAS_XGB else 'No'}")
        st.write(f"SMOTE disponible: {'Sí' if HAS_SMOTE else 'No'}")
        st.write(f"Registros salud: {len(health['df'])}")
        st.write(f"Registros inventario: {len(inventory['df'])}")

    if page == "Resumen":
        st.subheader("Resumen ejecutivo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline Salud F1", f"{health['baseline_f1']:.3f}")
        c2.metric("Baseline Stock 7d MAE", f"{inventory['baseline_mae_7']:.1f}")
        c3.metric("Baseline Stock 14d MAE", f"{inventory['baseline_mae_14']:.1f}")
        c4.metric("ODS foco", "ODS 3 / ODS 10")

        fig, ax = plt.subplots(figsize=(10, 4.8))
        labels = ["Árbol Baseline", "Linear 7d", "Linear 14d"]
        values = [health["baseline_f1"], inventory["baseline_r2_7"], inventory["baseline_r2_14"]]
        ax.bar(labels, values, color=["lightcoral", "steelblue", "mediumseagreen"], edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_title("Indicadores base del proyecto")
        st.pyplot(fig, clear_figure=True)
        st.write("La app ya reproduce el preprocessing del notebook y deja lista la capa de visualización para el demo.")

    elif page == "Salud":
        health_models = train_health_models(health["X_train"], health["y_train"], health["X_test"], health["y_test"], health["baseline_f1"])
        health_models.update({"dt_base": health["dt_base"], "feature_cols": health["feature_cols"]})
        health_section(health["df"], health["encoder"], health["X_test"], health["y_test"], health_models)

    elif page == "Logística":
        inventory_models = train_inventory_models(
            inventory["X_train"],
            inventory["X_test"],
            inventory["y_train_7"],
            inventory["y_test_7"],
            inventory["y_train_14"],
            inventory["y_test_14"],
        )
        inventory_models.update({"baseline_mae_7": inventory["baseline_mae_7"], "baseline_mae_14": inventory["baseline_mae_14"]})
        inventory_section(inventory["df"], inventory["X_test"], inventory["y_test_7"], inventory["y_test_14"], inventory_models)

    else:
        st.subheader("Datos y cobertura")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Salud")
            st.dataframe(health["df"].head(12), use_container_width=True)
        with col2:
            st.markdown("#### Inventario")
            st.dataframe(inventory["df"].head(12), use_container_width=True)


if __name__ == "__main__":
    main()