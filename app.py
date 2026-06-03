import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import shap
import warnings
warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ROP Prediction Dashboard",
    page_icon="⛏️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #111827;
}
[data-testid="stSidebar"] * {
    color: white;
}
.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-title {
    font-size: 18px;
    color: #6b7280;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #111827;
}
.block-container {
    padding-top: 1.5rem;
}
.chart-box {
    background: white;
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("⛏️ ROP Prediction Professional Dashboard")
st.markdown("Upload drilling data, explore insights, train models, evaluate predictions, and interpret results.")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# =========================
# READ FILE
# =========================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# =========================
# OUTLIER FUNCTIONS
# =========================
def remove_outliers_iqr(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) &
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]
    return df_clean

def remove_outliers_zscore(df, cols, threshold=3):
    df_clean = df.copy()
    for col in cols:
        z = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean = df_clean[np.abs(z) < threshold]
    return df_clean

# =========================
# MODEL BUILDER
# =========================
def get_model_and_params(model_name):
    if model_name == "Linear Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        params = {}
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
        params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5]
        }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
        params = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5]
        }
    elif model_name == "Extra Trees":
        model = ExtraTreesRegressor(random_state=42)
        params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5]
        }
    return model, params

# =========================
# MAIN APP
# =========================
if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Your dataset must have at least two numeric columns.")
        st.stop()

    target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)
    feature_cols = st.sidebar.multiselect(
        "Select Feature Columns",
        [c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col][:5]
    )

    depth_col = st.sidebar.selectbox("Select Depth/Index Column", df.columns)

    outlier_method = st.sidebar.selectbox(
        "Outlier Handling",
        ["None", "IQR", "Z-Score"]
    )

    model_name = st.sidebar.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest", "Gradient Boosting", "Extra Trees"]
    )

    tuning_method = st.sidebar.selectbox(
        "Hyperparameter Tuning",
        ["None", "GridSearchCV", "RandomizedSearchCV"]
    )

    use_shap = st.sidebar.checkbox("Enable SHAP Explainability", value=False)

    run_button = st.sidebar.button("🚀 Run Model")

    if run_button:
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
            st.stop()

        data = df[[depth_col] + feature_cols + [target_col]].dropna().copy()

        before_rows = len(data)

        if outlier_method == "IQR":
            data = remove_outliers_iqr(data, feature_cols + [target_col])
        elif outlier_method == "Z-Score":
            data = remove_outliers_zscore(data, feature_cols + [target_col])

        after_rows = len(data)

        X = data[feature_cols]
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model, param_grid = get_model_and_params(model_name)

        if tuning_method == "GridSearchCV" and len(param_grid) > 0:
            search = GridSearchCV(
                model,
                param_grid,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
        elif tuning_method == "RandomizedSearchCV" and len(param_grid) > 0:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=5,
                cv=3,
                scoring="r2",
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # =========================
        # METRICS
        # =========================
        st.subheader("Model Performance")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rows Before</div>
                <div class="metric-value">{before_rows}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rows After</div>
                <div class="metric-value">{after_rows}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R² Score</div>
                <div class="metric-value">{r2:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">RMSE</div>
                <div class="metric-value">{rmse:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        m5, m6 = st.columns(2)

        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE</div>
                <div class="metric-value">{mae:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Model</div>
                <div class="metric-value" style="font-size:20px;">{model_name}</div>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # EDA SECTION
        # =========================
        st.subheader("Exploratory Data Analysis")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            fig_hist = px.histogram(data, x=target_col, title=f"Distribution of {target_col}")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            corr = data[feature_cols + [target_col]].corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # PREDICTION PLOTS
        # =========================
        st.subheader("Prediction Dashboard")

        col1, col2 = st.columns(2)

        # Feature importance
        with col1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown("### Feature Importance")

            try:
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    fi_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)
                    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
                    st.plotly_chart(fig_fi, use_container_width=True)
                elif hasattr(best_model, "named_steps"):
                    lr_model = best_model.named_steps["model"]
                    if hasattr(lr_model, "coef_"):
                        coef_df = pd.DataFrame({
                            "Feature": feature_cols,
                            "Importance": np.abs(lr_model.coef_)
                        }).sort_values(by="Importance", ascending=False)
                        fig_coef = px.bar(coef_df, x="Importance", y="Feature", orientation="h")
                        st.plotly_chart(fig_coef, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model.")
                else:
                    st.info("Feature importance not available for this model.")
            except:
                st.info("Could not generate feature importance chart.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Actual vs Predicted
        with col2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown("### Actual vs Predicted")
            results_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred
            })
            fig_scatter = px.scatter(
                results_df, x="Actual", y="Predicted",
                trendline="ols",
                title="Actual vs Predicted"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Depth trend plot
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown("### Depth Trend Plot")

        trend_df = data.loc[X_test.index, [depth_col]].copy()
        trend_df["Actual"] = y_test
        trend_df["Predicted"] = y_pred
        trend_df = trend_df.sort_values(by=depth_col)

        fig_trend = px.line(
            trend_df,
            x=depth_col,
            y=["Actual", "Predicted"],
            title=f"{target_col} Trend vs {depth_col}"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Residual plot
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown("### Residual Analysis")
        residuals = y_test - y_pred
        fig_res = px.histogram(
            x=residuals,
            nbins=40,
            title="Residual Distribution"
        )
        fig_res.update_layout(xaxis_title="Residuals", yaxis_title="Count")
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # SHAP SECTION
        # =========================
        if use_shap:
            st.subheader("SHAP Explainability")

            try:
                shap_model = best_model
                shap_X = X_test.copy()

                if hasattr(best_model, "named_steps"):
                    st.info("SHAP is more reliable for tree-based models. Skipping pipeline-based SHAP.")
                else:
                    explainer = shap.Explainer(shap_model, shap_X)
                    shap_values = explainer(shap_X)

                    s1, s2 = st.columns(2)

                    with s1:
                        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                        st.markdown("### SHAP Summary Plot")
                        fig_shap1, ax1 = plt.subplots()
                        shap.summary_plot(shap_values, shap_X, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_shap1)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with s2:
                        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                        st.markdown("### SHAP Bar Plot")
                        fig_shap2, ax2 = plt.subplots()
                        shap.summary_plot(shap_values, shap_X, plot_type="bar", show=False)
                        plt.tight_layout()
                        st.pyplot(fig_shap2)
                        st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"SHAP could not be generated: {e}")

        # =========================
        # DOWNLOAD RESULTS
        # =========================
        st.subheader("Download Results")
        download_df = trend_df.copy()
        csv = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name="rop_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV or Excel file from the sidebar to begin.")
