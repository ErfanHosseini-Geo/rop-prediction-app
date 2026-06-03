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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="ROP Prediction App", layout="wide")

st.title("ROP Prediction Dashboard")
st.markdown("Upload your Excel/CSV file and predict **ROP** using machine learning models.")


# -----------------------------
# Helpers
# -----------------------------
def make_unique_columns(columns):
    """
    Make duplicate/empty column names unique.
    """
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
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    df.columns = make_unique_columns(df.columns)
    return df


def get_model(model_name, random_state, n_estimators, max_depth):
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else 3,
            random_state=random_state
        )
    elif model_name == "Linear Regression":
        return LinearRegression()
    else:
        raise ValueError("Unknown model selected.")


def safe_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel or CSV file",
    type=["csv", "xlsx", "xls"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Extra Trees", "Gradient Boosting", "Linear Regression"]
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, value=42, step=1)

if model_name != "Linear Regression":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 30, 10, 1)
else:
    n_estimators = 100
    max_depth = 0

run_button = st.sidebar.button("Run Model")


# -----------------------------
# Main
# -----------------------------
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        # Check target
        if "ROP" not in df.columns:
            st.error("Column 'ROP' was not found in your uploaded file.")
            st.stop()

        # Default columns to exclude
        exclude_cols_default = []
        for c in ["ROP", "WELL", "GROUP", "FORMATION", "FORCE_2020_LITHOFACIES_LITHOLOGY"]:
            if c in df.columns:
                exclude_cols_default.append(c)

        st.subheader("Feature Selection")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        candidate_features = [c for c in numeric_cols if c != "ROP"]

        selected_features = st.multiselect(
            "Select input features",
            options=candidate_features,
            default=candidate_features
        )

        depth_col = "DEPTH_MD" if "DEPTH_MD" in df.columns else None

        if len(selected_features) == 0:
            st.warning("Please select at least one feature.")
            st.stop()

        model_df = df[selected_features + ["ROP"]].copy()

        # Remove rows where target is missing
        model_df = model_df.dropna(subset=["ROP"])

        if model_df.shape[0] < 10:
            st.error("Not enough rows after removing missing target values.")
            st.stop()

        X = model_df[selected_features]
        y = model_df["ROP"]

        if run_button:
            model = get_model(model_name, random_state, n_estimators, max_depth)

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model)
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = safe_rmse(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.subheader("Model Performance")

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("MAE", f"{mae:.4f}")

            # Prediction table
            results_df = X_test.copy()
            results_df["Actual_ROP"] = y_test.values
            results_df["Predicted_ROP"] = y_pred
            if depth_col is not None and depth_col in df.columns and depth_col in results_df.columns:
                pass

            st.subheader("Prediction Results")
            st.dataframe(results_df.head(20))

            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_out,
                file_name="rop_predictions.csv",
                mime="text/csv"
            )

            # -----------------------------
            # Plots
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Actual vs Predicted")
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                ax1.scatter(y_test, y_pred, alpha=0.7)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax1.plot([min_val, max_val], [min_val, max_val], "r--")
                ax1.set_xlabel("Actual ROP")
                ax1.set_ylabel("Predicted ROP")
                ax1.set_title("Actual vs Predicted")
                st.pyplot(fig1)

            with col2:
                st.subheader("Residual Distribution")
                residuals = y_test - y_pred
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.hist(residuals, bins=30, edgecolor="black")
                ax2.set_xlabel("Residual (Actual - Predicted)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Residual Distribution")
                st.pyplot(fig2)

            # Depth plot if DEPTH_MD selected
            if depth_col is not None and depth_col in selected_features:
                st.subheader("ROP vs Depth")

                depth_test = X_test[depth_col].copy()
                depth_plot_df = pd.DataFrame({
                    "DEPTH_MD": depth_test,
                    "Actual_ROP": y_test,
                    "Predicted_ROP": y_pred
                }).sort_values("DEPTH_MD")

                fig3, ax3 = plt.subplots(figsize=(8, 8))
                ax3.plot(depth_plot_df["Actual_ROP"], depth_plot_df["DEPTH_MD"], label="Actual ROP")
                ax3.plot(depth_plot_df["Predicted_ROP"], depth_plot_df["DEPTH_MD"], label="Predicted ROP")
                ax3.invert_yaxis()
                ax3.set_xlabel("ROP")
                ax3.set_ylabel("Depth")
                ax3.set_title("ROP vs Depth")
                ax3.legend()
                st.pyplot(fig3)

            # Feature importance
            final_model = pipeline.named_steps["model"]
            if hasattr(final_model, "feature_importances_"):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Importance": final_model.feature_importances_
                }).sort_values("Importance", ascending=False)

                fig4, ax4 = plt.subplots(figsize=(8, 5))
                ax4.barh(importance_df["Feature"], importance_df["Importance"])
                ax4.invert_yaxis()
                ax4.set_xlabel("Importance")
                ax4.set_title("Feature Importance")
                st.pyplot(fig4)

                st.dataframe(importance_df)

            elif hasattr(final_model, "coef_"):
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Coefficient": final_model.coef_
                }).sort_values("Coefficient", ascending=False)

                fig5, ax5 = plt.subplots(figsize=(8, 5))
                ax5.barh(coef_df["Feature"], coef_df["Coefficient"])
                ax5.invert_yaxis()
                ax5.set_xlabel("Coefficient")
                ax5.set_title("Linear Model Coefficients")
                st.pyplot(fig5)

                st.dataframe(coef_df)

    except Exception as e:
        st.error(f"Error while processing file: {e}")

else:
    st.info("Please upload an Excel or CSV file from the sidebar to begin.")
