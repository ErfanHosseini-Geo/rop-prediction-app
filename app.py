import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="ROP Well-Based Prediction", layout="wide")
st.title("ROP Prediction Dashboard - Development Well + Blind Well")
st.markdown("Train the model on one well and evaluate on another unseen well.")


# ---------------------------------
# Helpers
# ---------------------------------
def make_unique_columns(columns):
    seen = {}
    new_cols = []

    for i, col in enumerate(columns):
        if col is None or str(col).strip() == "" or str(col).startswith("Unnamed"):
            col = f"Unnamed_{i}"

        col = str(col).strip()

        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)

    return new_cols


@st.cache_data
def load_data(uploaded_file):
    fname = uploaded_file.name.lower()

    if fname.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif fname.endswith(".xlsx") or fname.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format.")

    df.columns = make_unique_columns(df.columns)
    return df


def build_model(model_name, random_state, n_estimators, max_depth):
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth == 0 else max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth == 0 else max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=3 if max_depth == 0 else max_depth,
            random_state=random_state
        )
    elif model_name == "Linear Regression":
        return LinearRegression()
    else:
        raise ValueError("Unknown model selected.")


def calc_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


def plot_actual_vs_pred(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], "r--")
    ax.set_xlabel("Actual ROP")
    ax.set_ylabel("Predicted ROP")
    ax.set_title(title)
    return fig


def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residuals, bins=30, edgecolor="black")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    return fig


def plot_depth_curve(results_df, depth_col, title):
    plot_df = results_df[[depth_col, "Actual_ROP", "Predicted_ROP"]].copy()
    plot_df = plot_df.sort_values(depth_col)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(plot_df["Actual_ROP"], plot_df[depth_col], label="Actual ROP")
    ax.plot(plot_df["Predicted_ROP"], plot_df[depth_col], label="Predicted ROP")
    ax.invert_yaxis()
    ax.set_xlabel("ROP")
    ax.set_ylabel(depth_col)
    ax.set_title(title)
    ax.legend()
    return fig


# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)

model_name = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Extra Trees", "Gradient Boosting", "Linear Regression"]
)

test_size = st.sidebar.slider("Internal Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, value=42, step=1)

if model_name != "Linear Regression":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.sidebar.slider("max_depth (0=None)", 0, 30, 10, 1)
else:
    n_estimators = 100
    max_depth = 0


# ---------------------------------
# Main
# ---------------------------------
if uploaded_file is None:
    st.info("Please upload your file from the sidebar.")
    st.stop()

try:
    df = load_data(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    # Replace -999 with NaN
    df = df.replace(-999, np.nan)

    if "ROP" not in df.columns:
        st.error("Column 'ROP' not found.")
        st.stop()

    if "WELL" not in df.columns:
        st.error("Column 'WELL' not found. This mode requires a WELL column.")
        st.stop()

    well_col = "WELL"
    depth_col = "DEPTH_MD" if "DEPTH_MD" in df.columns else None

    wells = sorted(df[well_col].dropna().astype(str).unique().tolist())

    if len(wells) < 2:
        st.error("At least two wells are required.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Well Split Setup")

    dev_well = st.sidebar.selectbox("Development Well (Train/Test)", wells, index=0)

    blind_options = [w for w in wells if w != dev_well]
    blind_well = st.sidebar.selectbox("Blind Well", blind_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Columns Setup")

    target_col = st.sidebar.selectbox(
        "Target Column",
        options=df.columns.tolist(),
        index=df.columns.tolist().index("ROP")
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    default_exclude = []
    for c in [target_col, "ROPA", "FORCE_2020_LITHOFACIES_LITHOLOGY"]:
        if c in numeric_cols:
            default_exclude.append(c)

    default_features = [c for c in numeric_cols if c not in default_exclude]

    selected_features = st.multiselect(
        "Select Input Features",
        options=numeric_cols,
        default=default_features
    )

    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    run_button = st.sidebar.button("Run Model")

    if not run_button:
        st.stop()

    # -----------------------------
    # Prepare data
    # -----------------------------
    needed_cols = [well_col, target_col] + selected_features
    if depth_col is not None and depth_col not in needed_cols:
        needed_cols.append(depth_col)

    model_df = df[needed_cols].copy()
    model_df = model_df.dropna(subset=[target_col])

    dev_df = model_df[model_df[well_col].astype(str) == str(dev_well)].copy()
    blind_df = model_df[model_df[well_col].astype(str) == str(blind_well)].copy()

    if dev_df.empty:
        st.error("Development well has no valid rows.")
        st.stop()

    if blind_df.empty:
        st.error("Blind well has no valid rows.")
        st.stop()

    X_dev = dev_df[selected_features]
    y_dev = dev_df[target_col]

    X_blind = blind_df[selected_features]
    y_blind = blind_df[target_col]

    # internal split only on development well
    X_train, X_test, y_train, y_test = train_test_split(
        X_dev, y_dev,
        test_size=test_size,
        random_state=random_state
    )

    test_meta = dev_df.loc[X_test.index].copy()
    blind_meta = blind_df.copy()

    # -----------------------------
    # Model
    # -----------------------------
    model = build_model(model_name, random_state, n_estimators, max_depth)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # predictions
    y_test_pred = pipeline.predict(X_test)
    y_blind_pred = pipeline.predict(X_blind)

    # metrics
    test_metrics = calc_metrics(y_test, y_test_pred)
    blind_metrics = calc_metrics(y_blind, y_blind_pred)

    # result tables
    test_results = test_meta.copy()
    test_results["Actual_ROP"] = y_test.values
    test_results["Predicted_ROP"] = y_test_pred
    test_results["Residual"] = test_results["Actual_ROP"] - test_results["Predicted_ROP"]

    blind_results = blind_meta.copy()
    blind_results["Actual_ROP"] = y_blind.values
    blind_results["Predicted_ROP"] = y_blind_pred
    blind_results["Residual"] = blind_results["Actual_ROP"] - blind_results["Predicted_ROP"]

    # -----------------------------
    # Summary
    # -----------------------------
    st.subheader("Run Summary")
    st.json({
        "Model": model_name,
        "Development Well": dev_well,
        "Blind Well": blind_well,
        "Target": target_col,
        "Features Count": len(selected_features),
        "Development Rows": len(dev_df),
        "Blind Rows": len(blind_df),
        "Train Rows": len(X_train),
        "Internal Test Rows": len(X_test)
    })

    # -----------------------------
    # Metrics
    # -----------------------------
    st.subheader("Internal Test Metrics (from Development Well)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{test_metrics['R2']:.4f}")
    c2.metric("RMSE", f"{test_metrics['RMSE']:.4f}")
    c3.metric("MAE", f"{test_metrics['MAE']:.4f}")

    st.subheader("Blind Test Metrics (Unseen Well)")
    c4, c5, c6 = st.columns(3)
    c4.metric("R²", f"{blind_metrics['R2']:.4f}")
    c5.metric("RMSE", f"{blind_metrics['RMSE']:.4f}")
    c6.metric("MAE", f"{blind_metrics['MAE']:.4f}")

    # -----------------------------
    # Plots - Internal Test
    # -----------------------------
    st.subheader("Internal Test Plots")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_actual_vs_pred(y_test, y_test_pred, "Internal Test: Actual vs Predicted"))

    with col2:
        st.pyplot(plot_residuals(y_test, y_test_pred, "Internal Test: Residual Distribution"))

    if depth_col is not None and depth_col in test_results.columns:
        st.pyplot(plot_depth_curve(test_results, depth_col, "Internal Test: ROP vs Depth"))

    # -----------------------------
    # Plots - Blind Test
    # -----------------------------
    st.subheader("Blind Test Plots")
    col3, col4 = st.columns(2)

    with col3:
        st.pyplot(plot_actual_vs_pred(y_blind, y_blind_pred, "Blind Test: Actual vs Predicted"))

    with col4:
        st.pyplot(plot_residuals(y_blind, y_blind_pred, "Blind Test: Residual Distribution"))

    if depth_col is not None and depth_col in blind_results.columns:
        st.pyplot(plot_depth_curve(blind_results, depth_col, "Blind Test: ROP vs Depth"))

    # -----------------------------
    # Feature importance
    # -----------------------------
    final_model = pipeline.named_steps["model"]

    if hasattr(final_model, "feature_importances_"):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": final_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        ax_imp.barh(importance_df["Feature"], importance_df["Importance"])
        ax_imp.invert_yaxis()
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)

        st.dataframe(importance_df)

    elif hasattr(final_model, "coef_"):
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": selected_features,
            "Coefficient": final_model.coef_
        }).sort_values("Coefficient", ascending=False)

        fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
        ax_coef.barh(coef_df["Feature"], coef_df["Coefficient"])
        ax_coef.invert_yaxis()
        ax_coef.set_xlabel("Coefficient")
        ax_coef.set_title("Linear Model Coefficients")
        st.pyplot(fig_coef)

        st.dataframe(coef_df)

    # -----------------------------
    # Tables
    # -----------------------------
    st.subheader("Internal Test Predictions")
    st.dataframe(test_results.head(50))

    st.subheader("Blind Test Predictions")
    st.dataframe(blind_results.head(50))

    # downloads
    st.download_button(
        "Download Internal Test Predictions CSV",
        data=test_results.to_csv(index=False).encode("utf-8"),
        file_name="internal_test_predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Blind Test Predictions CSV",
        data=blind_results.to_csv(index=False).encode("utf-8"),
        file_name="blind_test_predictions.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error: {e}")
