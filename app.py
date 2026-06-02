import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats import zscore
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    IsolationForest
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ROP Prediction Pro", page_icon="⛏️", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.stButton>button, .stDownloadButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("⛏️ ROP Prediction Professional Dashboard")
st.markdown("Upload drilling data, perform EDA, remove outliers, tune models automatically, and predict ROP.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    st.info("Please upload a dataset from the sidebar to begin.")
    st.stop()

# ---------------- LOAD DATA ----------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset Overview",
    "EDA",
    "Outlier Detection",
    "Model Training",
    "Explainability & Downloads"
])

# =========================================================
# TAB 1 - DATASET OVERVIEW
# =========================================================
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Count": df.isna().sum().values,
        "Missing %": np.round((df.isna().sum().values / len(df)) * 100, 2)
    })
    st.markdown("### Data Types and Missing Values")
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# =========================================================
# TAB 2 - EDA
# =========================================================
with tab2:
    st.subheader("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown("### Missing Value Heatmap")
    if df.isna().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Values Heatmap")
        st.pyplot(fig)
    else:
        st.success("No missing values found in the dataset.")

    if len(numeric_cols) > 0:
        st.markdown("### Distribution Plot")
        selected_num = st.selectbox("Select a numerical feature", numeric_cols, key="eda_num")
        fig = px.histogram(df, x=selected_num, nbins=40, title=f"Distribution of {selected_num}")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Boxplot")
        fig_box = px.box(df, y=selected_num, title=f"Boxplot of {selected_num}")
        st.plotly_chart(fig_box, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.markdown("### Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Correlation Heatmap")
        st.pyplot(fig_corr)

        st.markdown("### Scatter Plot")
        x_scatter = st.selectbox("X-axis feature", numeric_cols, key="scatter_x")
        y_scatter = st.selectbox("Y-axis feature", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
        fig_scatter = px.scatter(df, x=x_scatter, y=y_scatter, title=f"{x_scatter} vs {y_scatter}")
        st.plotly_chart(fig_scatter, use_container_width=True)

    if len(categorical_cols) > 0:
        st.markdown("### Categorical Summary")
        selected_cat = st.selectbox("Select a categorical feature", categorical_cols, key="eda_cat")
        cat_counts = df[selected_cat].astype(str).value_counts().reset_index()
        cat_counts.columns = [selected_cat, "Count"]
        fig_cat = px.bar(cat_counts, x=selected_cat, y="Count", title=f"Category Counts: {selected_cat}")
        st.plotly_chart(fig_cat, use_container_width=True)

# =========================================================
# TAB 3 - OUTLIER DETECTION
# =========================================================
with tab3:
    st.subheader("Outlier Detection")

    working_df = df.copy()

    if len(df.select_dtypes(include=[np.number]).columns) == 0:
        st.warning("No numerical columns available for outlier detection.")
        cleaned_df = working_df.copy()
    else:
        outlier_method = st.selectbox("Select Outlier Detection Method", ["None", "IQR", "Z-score", "Isolation Forest"])
        remove_outliers = st.checkbox("Remove detected outliers before modeling", value=False)

        numeric_cols_out = working_df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_mask = pd.Series([False] * len(working_df), index=working_df.index)

        if outlier_method == "IQR":
            selected_iqr_col = st.selectbox("Select numerical column for IQR", numeric_cols_out)
            q1 = working_df[selected_iqr_col].quantile(0.25)
            q3 = working_df[selected_iqr_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (working_df[selected_iqr_col] < lower) | (working_df[selected_iqr_col] > upper)

            st.write(f"Lower bound: {lower:.4f}")
            st.write(f"Upper bound: {upper:.4f}")

            fig_iqr = px.scatter(
                working_df.reset_index(),
                x=working_df.index,
                y=selected_iqr_col,
                color=outlier_mask.astype(str),
                title=f"IQR Outlier Detection - {selected_iqr_col}",
                labels={"color": "Outlier"}
            )
            st.plotly_chart(fig_iqr, use_container_width=True)

        elif outlier_method == "Z-score":
            selected_z_col = st.selectbox("Select numerical column for Z-score", numeric_cols_out)
            z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
            z_vals = np.abs(zscore(working_df[selected_z_col].dropna()))
            temp_mask = pd.Series([False] * len(working_df[selected_z_col].dropna()), index=working_df[selected_z_col].dropna().index)
            temp_mask[temp_mask.index] = z_vals > z_thresh
            outlier_mask = pd.Series([False] * len(working_df), index=working_df.index)
            outlier_mask.loc[temp_mask.index] = temp_mask

            fig_z = px.scatter(
                working_df.reset_index(),
                x=working_df.index,
                y=selected_z_col,
                color=outlier_mask.astype(str),
                title=f"Z-score Outlier Detection - {selected_z_col}",
                labels={"color": "Outlier"}
            )
            st.plotly_chart(fig_z, use_container_width=True)

        elif outlier_method == "Isolation Forest":
            contamination = st.slider("Contamination", 0.01, 0.20, 0.05, 0.01)
            iso_data = working_df[numeric_cols_out].copy()
            iso_data = iso_data.fillna(iso_data.median())
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(iso_data)
            outlier_mask = pd.Series(preds == -1, index=working_df.index)

            vis_col = st.selectbox("Select numerical column to visualize", numeric_cols_out, key="iso_vis")
            fig_iso = px.scatter(
                working_df.reset_index(),
                x=working_df.index,
                y=vis_col,
                color=outlier_mask.astype(str),
                title=f"Isolation Forest Outlier Detection - {vis_col}",
                labels={"color": "Outlier"}
            )
            st.plotly_chart(fig_iso, use_container_width=True)

        outlier_count = int(outlier_mask.sum()) if outlier_method != "None" else 0
        st.metric("Detected Outliers", outlier_count)

        if remove_outliers and outlier_method != "None":
            cleaned_df = working_df.loc[~outlier_mask].copy()
            st.success(f"Outliers removed. Cleaned dataset shape: {cleaned_df.shape}")
        else:
            cleaned_df = working_df.copy()

        st.markdown("### Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(20), use_container_width=True)

# =========================================================
# TAB 4 - MODEL TRAINING
# =========================================================
with tab4:
    st.subheader("Model Training and Automatic Hyperparameter Tuning")

    if cleaned_df.empty:
        st.warning("The cleaned dataset is empty. Please adjust outlier settings.")
        st.stop()

    all_cols = cleaned_df.columns.tolist()
    target = st.selectbox("Select Target Column", all_cols)

    features = st.multiselect(
        "Select Input Features",
        [c for c in all_cols if c != target],
        default=[c for c in all_cols if c != target]
    )

    if len(features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    data = cleaned_df[features + [target]].copy()
    data = data.dropna(subset=[target])

    X = data[features].copy()
    y = data[target].copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5)
    with c2:
        random_state = st.number_input("Random State", value=42, step=1)
    with c3:
        scale_numeric = st.checkbox("Scale Numerical Features", value=True)

    model_list = [
        "Linear Regression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "AdaBoost",
        "SVR",
        "KNN"
    ]
    if HAS_XGB:
        model_list.append("XGBoost")

    model_name = st.selectbox("Select Regression Model", model_list)

    tuning_method = st.selectbox("Hyperparameter Tuning Method", ["None", "Grid Search", "Random Search"])
    scoring_metric = st.selectbox("Scoring Metric", ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"])
    cv_folds = st.slider("Cross Validation Folds", 3, 10, 5, 1)
    n_iter_random = st.slider("Random Search Iterations", 5, 50, 15, 1)

    numeric_transformer_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_transformer_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    # Base model + param grids
    param_grid = {}

    if model_name == "Linear Regression":
        model = LinearRegression()
        param_grid = {}

    elif model_name == "Ridge":
        model = Ridge()
        param_grid = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
        }

    elif model_name == "Lasso":
        model = Lasso()
        param_grid = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }

    elif model_name == "ElasticNet":
        model = ElasticNet()
        param_grid = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }

    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(random_state=random_state)
        param_grid = {
            "model__max_depth": [3, 5, 8, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }

    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }

    elif model_name == "Extra Trees":
        model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }

    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=random_state)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 5, 8]
        }

    elif model_name == "AdaBoost":
        model = AdaBoostRegressor(random_state=random_state)
        param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
        }

    elif model_name == "SVR":
        model = SVR()
        param_grid = {
            "model__C": [0.1, 1, 10, 50, 100],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
            "model__kernel": ["rbf", "linear", "poly"]
        }

    elif model_name == "KNN":
        model = KNeighborsRegressor()
        param_grid = {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }

    elif model_name == "XGBoost" and HAS_XGB:
        model = XGBRegressor(random_state=random_state, n_jobs=-1)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 10],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2]
        }

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if st.button("Train Model"):
        if len(data) < 10:
            st.warning("Dataset is too small. At least 10 rows are recommended.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=random_state
        )

        best_model = pipe
        best_params = {}
        best_cv_score = None

        with st.spinner("Training model... please wait."):
            if tuning_method == "None" or model_name == "Linear Regression" or len(param_grid) == 0:
                best_model.fit(X_train, y_train)

            elif tuning_method == "Grid Search":
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring=scoring_metric,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_cv_score = search.best_score_

            elif tuning_method == "Random Search":
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_grid,
                    n_iter=n_iter_random,
                    cv=cv_folds,
                    scoring=scoring_metric,
                    random_state=random_state,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_cv_score = search.best_score_

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.markdown("## Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("MAE", f"{mae:.4f}")

        if tuning_method != "None" and len(best_params) > 0:
            st.markdown("### Best Hyperparameters")
            st.json(best_params)
            if best_cv_score is not None:
                st.write(f"Best CV Score: {best_cv_score:.4f}")

        results = pd.DataFrame({
            "Actual_ROP": y_test.values,
            "Predicted_ROP": y_pred,
            "Residual": y_test.values - y_pred
        })

        st.markdown("## Prediction Results")
        st.dataframe(results.head(30), use_container_width=True)

        colp1, colp2 = st.columns(2)

        with colp1:
            fig1 = px.scatter(
                results,
                x="Actual_ROP",
                y="Predicted_ROP",
                title="Actual vs Predicted"
            )
            fig1.add_shape(
                type="line",
                x0=results["Actual_ROP"].min(),
                y0=results["Actual_ROP"].min(),
                x1=results["Actual_ROP"].max(),
                y1=results["Actual_ROP"].max(),
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig1, use_container_width=True)

        with colp2:
            fig2 = px.scatter(
                results,
                x="Predicted_ROP",
                y="Residual",
                title="Residual Plot"
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

        colp3, colp4 = st.columns(2)

        with colp3:
            fig3 = px.histogram(results, x="Residual", nbins=30, title="Residual Distribution")
            st.plotly_chart(fig3, use_container_width=True)

        with colp4:
            sample_n = min(50, len(results))
            sample_df = results.head(sample_n).reset_index(drop=True)
            fig4 = px.line(
                sample_df.reset_index(),
                x="index",
                y=["Actual_ROP", "Predicted_ROP"],
                title="Actual vs Predicted Trend"
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Save for tab 5
        st.session_state["best_model"] = best_model
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["results"] = results
        st.session_state["num_cols"] = num_cols
        st.session_state["cat_cols"] = cat_cols
        st.session_state["features"] = features
        st.session_state["model_name"] = model_name
        st.session_state["cleaned_df"] = cleaned_df

# =========================================================
# TAB 5 - EXPLAINABILITY & DOWNLOADS
# =========================================================
with tab5:
    st.subheader("Explainability and Downloads")

    if "results" not in st.session_state:
        st.info("Train a model first to view explainability and download outputs.")
    else:
        best_model = st.session_state["best_model"]
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        results = st.session_state["results"]
        num_cols = st.session_state["num_cols"]
        cat_cols = st.session_state["cat_cols"]
        features = st.session_state["features"]
        model_name = st.session_state["model_name"]
        cleaned_df = st.session_state["cleaned_df"]

        # Feature importance
        if model_name in ["Random Forest", "Extra Trees", "XGBoost", "Decision Tree", "Gradient Boosting"]:
            try:
                fitted_model = best_model.named_steps["model"]
                transformed_feature_names = []

                transformed_feature_names.extend(num_cols)

                if len(cat_cols) > 0:
                    ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
                    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
                    transformed_feature_names.extend(ohe_names)

                importances = fitted_model.feature_importances_
                imp_df = pd.DataFrame({
                    "Feature": transformed_feature_names[:len(importances)],
                    "Importance": importances
                }).sort_values("Importance", ascending=False)

                st.markdown("### Feature Importance")
                fig_imp = px.bar(
                    imp_df.head(20),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 20 Feature Importances"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                st.dataframe(imp_df.head(20), use_container_width=True)

            except Exception as e:
                st.warning(f"Could not display feature importance: {e}")

        # SHAP
        if HAS_SHAP and model_name in ["Random Forest", "Extra Trees", "XGBoost", "Decision Tree", "Gradient Boosting"]:
            try:
                st.markdown("### SHAP Explainability")

                X_train_transformed = best_model.named_steps["preprocessor"].transform(X_train)
                X_test_transformed = best_model.named_steps["preprocessor"].transform(X_test)

                transformed_feature_names = []
                transformed_feature_names.extend(num_cols)

                if len(cat_cols) > 0:
                    ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
                    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
                    transformed_feature_names.extend(ohe_names)

                X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names)

                explainer = shap.Explainer(best_model.named_steps["model"], X_train_transformed)
                shap_values = explainer(X_test_transformed)

                st.markdown("#### SHAP Summary Plot")
                fig_shap1 = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_transformed_df, show=False)
                st.pyplot(fig_shap1, clear_figure=True)

                st.markdown("#### SHAP Bar Plot")
                fig_shap2 = plt.figure(figsize=(10, 6))
                shap.plots.bar(shap_values, show=False)
                st.pyplot(fig_shap2, clear_figure=True)

            except Exception as e:
                st.warning(f"SHAP plots could not be generated: {e}")
        elif not HAS_SHAP:
            st.info("SHAP is not installed. Add 'shap' to requirements.txt.")

        st.markdown("### Downloads")

        pred_csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prediction Results",
            pred_csv,
            "rop_predictions.csv",
            "text/csv"
        )

        clean_csv = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned Dataset",
            clean_csv,
            "cleaned_dataset.csv",
            "text/csv"
        )

        metrics_df = pd.DataFrame([{
            "Model": model_name,
            "R2": r2_score(st.session_state["y_test"], st.session_state["results"]["Predicted_ROP"]),
            "RMSE": np.sqrt(mean_squared_error(st.session_state["y_test"], st.session_state["results"]["Predicted_ROP"])),
            "MAE": mean_absolute_error(st.session_state["y_test"], st.session_state["results"]["Predicted_ROP"])
        }])
        metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Metrics",
            metrics_csv,
            "model_metrics.csv",
            "text/csv"
        )
