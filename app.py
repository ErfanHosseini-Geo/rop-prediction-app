import io
import joblib
import shap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


st.set_page_config(page_title="Advanced ROP Dashboard", layout="wide")
st.title("Advanced ROP Prediction Dashboard")
st.markdown("Development Well -> Train/Test | Blind Well -> Final Unseen Evaluation")


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


def unique_list(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = make_unique_columns(df.columns)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


def get_model(name, rs=42):
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=rs,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=rs,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=rs,
        ),
        "Linear Regression": LinearRegression(),
    }
    return models[name]


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def safe_depth_sort_plot(ax, df_plot, depth_col, actual_col, pred_col=None, title=""):
    cols = [depth_col, actual_col]
    if pred_col is not None:
        cols.append(pred_col)

    cols = unique_list(cols)
    temp = df_plot[cols].dropna().sort_values(depth_col)

    ax.plot(temp[actual_col], temp[depth_col], label="Actual", linewidth=1.8)
    if pred_col is not None:
        ax.plot(temp[pred_col], temp[depth_col], label="Predicted", linewidth=1.6)

    ax.invert_yaxis()
    ax.set_xlabel("ROP")
    ax.set_ylabel(depth_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


class DataCleaner:
    def __init__(
        self,
        numeric_cols,
        categorical_cols,
        missing_method="median",
        outlier_method="none",
        z_thresh=3.0,
        iqr_factor=1.5,
        iso_contamination=0.03,
    ):
        self.numeric_cols = unique_list(list(numeric_cols))
        self.categorical_cols = unique_list(list(categorical_cols))
        self.categorical_cols = [c for c in self.categorical_cols if c not in self.numeric_cols]

        self.missing_method = missing_method
        self.outlier_method = outlier_method
        self.z_thresh = z_thresh
        self.iqr_factor = iqr_factor
        self.iso_contamination = iso_contamination

        self.preprocessor = None
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.iso_model = None
        self.report_ = {}

    def _build_preprocessor(self):
        if self.missing_method == "median":
            num_imputer = SimpleImputer(strategy="median")
        elif self.missing_method == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        elif self.missing_method == "most_frequent":
            num_imputer = SimpleImputer(strategy="most_frequent")
        elif self.missing_method == "knn":
            num_imputer = KNNImputer(n_neighbors=5)
        else:
            num_imputer = SimpleImputer(strategy="median")

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        transformers = []

        if len(self.numeric_cols) > 0:
            transformers.append(("num", num_imputer, self.numeric_cols))

        if len(self.categorical_cols) > 0:
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def fit(self, X):
        X = X.copy()
        X = X.loc[:, ~pd.Index(X.columns).duplicated()].copy()

        expected_cols = unique_list(self.numeric_cols + self.categorical_cols)
        X = X[expected_cols].copy()

        self._build_preprocessor()

        missing_before = int(X.isna().sum().sum())

        X_imp = self.preprocessor.fit_transform(X)
        all_cols = unique_list(self.numeric_cols + self.categorical_cols)
        X_imp_df = pd.DataFrame(X_imp, columns=all_cols, index=X.index)

        missing_after = int(X_imp_df.isna().sum().sum())

        self.report_["missing_before_train"] = missing_before
        self.report_["missing_after_train"] = missing_after
        self.report_["train_rows_before_outlier"] = int(len(X_imp_df))

        mask = pd.Series(True, index=X_imp_df.index)

        if self.outlier_method == "iqr_capping":
            for col in self.numeric_cols:
                q1 = X_imp_df[col].quantile(0.25)
                q3 = X_imp_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_factor * iqr
                upper = q3 + self.iqr_factor * iqr
                self.lower_bounds_[col] = lower
                self.upper_bounds_[col] = upper

        elif self.outlier_method == "iqr_remove":
            for col in self.numeric_cols:
                q1 = X_imp_df[col].quantile(0.25)
                q3 = X_imp_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_factor * iqr
                upper = q3 + self.iqr_factor * iqr
                self.lower_bounds_[col] = lower
                self.upper_bounds_[col] = upper
                mask &= X_imp_df[col].between(lower, upper)

        elif self.outlier_method == "zscore_remove":
            if len(self.numeric_cols) > 0:
                zscores = np.abs(zscore(X_imp_df[self.numeric_cols], nan_policy="omit"))
                if len(self.numeric_cols) == 1:
                    mask &= pd.Series(zscores < self.z_thresh, index=X_imp_df.index)
                else:
                    mask &= pd.Series((zscores < self.z_thresh).all(axis=1), index=X_imp_df.index)

        elif self.outlier_method == "isolation_forest":
            if len(self.numeric_cols) > 0:
                self.iso_model = IsolationForest(
                    contamination=self.iso_contamination,
                    random_state=42,
                )
                preds = self.iso_model.fit_predict(X_imp_df[self.numeric_cols])
                mask &= pd.Series(preds == 1, index=X_imp_df.index)

        self.report_["train_rows_after_outlier"] = int(mask.sum())
        self.report_["train_rows_removed_outlier"] = int((~mask).sum())

        return self

    def transform(self, X, apply_row_removal=False):
        X = X.copy()
        X = X.loc[:, ~pd.Index(X.columns).duplicated()].copy()

        all_cols = unique_list(self.numeric_cols + self.categorical_cols)
        X = X[all_cols].copy()

        X_imp = self.preprocessor.transform(X)
        X_df = pd.DataFrame(X_imp, columns=all_cols, index=X.index)

        if self.outlier_method == "iqr_capping":
            for col in self.numeric_cols:
                lower = self.lower_bounds_.get(col, None)
                upper = self.upper_bounds_.get(col, None)
                if lower is not None and upper is not None:
                    X_df[col] = X_df[col].clip(lower=lower, upper=upper)

        elif self.outlier_method == "iqr_remove" and apply_row_removal:
            mask = pd.Series(True, index=X_df.index)
            for col in self.numeric_cols:
                lower = self.lower_bounds_.get(col, None)
                upper = self.upper_bounds_.get(col, None)
                if lower is not None and upper is not None:
                    mask &= X_df[col].between(lower, upper)
            X_df = X_df.loc[mask]

        elif self.outlier_method == "zscore_remove" and apply_row_removal:
            if len(self.numeric_cols) > 0:
                zscores = np.abs(zscore(X_df[self.numeric_cols], nan_policy="omit"))
                if len(self.numeric_cols) == 1:
                    mask = pd.Series(zscores < self.z_thresh, index=X_df.index)
                else:
                    mask = pd.Series((zscores < self.z_thresh).all(axis=1), index=X_df.index)
                X_df = X_df.loc[mask]

        elif self.outlier_method == "isolation_forest" and apply_row_removal:
            if self.iso_model is not None and len(self.numeric_cols) > 0:
                preds = self.iso_model.predict(X_df[self.numeric_cols])
                X_df = X_df.loc[pd.Series(preds == 1, index=X_df.index)]

        return X_df


uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df = df.replace(-999, np.nan)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Columns")
    st.write(df.columns.tolist())

    well_col = st.sidebar.selectbox("Select Well Column", df.columns)
    target_col = st.sidebar.selectbox("Select Target Column (ROP)", df.columns)

    possible_depth_cols = [c for c in df.columns if "depth" in c.lower()]
    if len(possible_depth_cols) > 0:
        default_depth = possible_depth_cols[0]
    else:
        default_depth = df.columns[0]

    depth_col = st.sidebar.selectbox(
        "Select Depth Column for Plotting",
        df.columns,
        index=list(df.columns).index(default_depth),
    )

    wells = df[well_col].dropna().astype(str).unique().tolist()
    wells = sorted(wells)

    dev_well = st.sidebar.selectbox("Select Development Well", wells)
    blind_options = [w for w in wells if w != dev_well]
    blind_well = st.sidebar.selectbox("Select Blind Well", blind_options)

    candidate_features = [c for c in df.columns if c not in [well_col, target_col]]
    selected_features = st.sidebar.multiselect(
        "Select Features",
        candidate_features,
        default=candidate_features[: min(8, len(candidate_features))],
    )

    model_names = st.sidebar.multiselect(
        "Select Models",
        ["Random Forest", "Extra Trees", "Gradient Boosting", "Linear Regression"],
        default=["Random Forest", "Extra Trees"],
    )

    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Cleaning")

    missing_method_label = st.sidebar.selectbox(
        "Missing Value Method",
        ["Median", "Mean", "Most Frequent", "KNN"],
    )
    missing_method_map = {
        "Median": "median",
        "Mean": "mean",
        "Most Frequent": "most_frequent",
        "KNN": "knn",
    }
    missing_method = missing_method_map[missing_method_label]

    outlier_method_label = st.sidebar.selectbox(
        "Outlier Method",
        ["None", "IQR Capping", "IQR Row Removal", "Z-Score Row Removal", "Isolation Forest"],
    )
    outlier_method_map = {
        "None": "none",
        "IQR Capping": "iqr_capping",
        "IQR Row Removal": "iqr_remove",
        "Z-Score Row Removal": "zscore_remove",
        "Isolation Forest": "isolation_forest",
    }
    outlier_method = outlier_method_map[outlier_method_label]

    z_thresh = st.sidebar.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
    iqr_factor = st.sidebar.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1)
    iso_contamination = st.sidebar.slider("Isolation Forest Contamination", 0.01, 0.20, 0.03, 0.01)

    run_button = st.sidebar.button("Run Modeling")

    if run_button:
        if len(selected_features) == 0:
            st.error("Please select at least one feature.")
            st.stop()

        if len(model_names) == 0:
            st.error("Please select at least one model.")
            st.stop()

        selected_features = unique_list(selected_features)
        selected_features = [c for c in selected_features if c not in [well_col, target_col]]
        selected_features = [c for c in selected_features if c in df.columns]

        if len(selected_features) == 0:
            st.error("No valid feature columns remain after removing duplicates and protected columns.")
            st.stop()

        base_cols = [well_col, target_col, depth_col] + selected_features
        final_cols = unique_list(base_cols)
        final_cols = [c for c in final_cols if c in df.columns]

        work_df = df[final_cols].copy()
        work_df = work_df.loc[:, ~pd.Index(work_df.columns).duplicated()].copy()

        dev_df = work_df[work_df[well_col].astype(str) == str(dev_well)].copy()
        blind_df = work_df[work_df[well_col].astype(str) == str(blind_well)].copy()

        dev_df = dev_df.dropna(subset=[target_col])
        blind_df = blind_df.dropna(subset=[target_col])

        if len(dev_df) < 10:
            st.error("Development well has too few rows after removing missing target values.")
            st.stop()

        if len(blind_df) < 3:
            st.warning("Blind well has too few rows. Results may be unstable.")

        X_dev = dev_df[selected_features].copy()
        y_dev = dev_df[target_col].copy()

        X_blind_raw = blind_df[selected_features].copy()
        y_blind = blind_df[target_col].copy()
        depth_blind = blind_df[depth_col].copy()

        X_dev = X_dev.loc[:, ~pd.Index(X_dev.columns).duplicated()].copy()
        X_blind_raw = X_blind_raw.loc[:, ~pd.Index(X_blind_raw.columns).duplicated()].copy()

        categorical_candidates = [c for c in ["FORMATION", "GROUP"] if c in selected_features]
        categorical_cols = unique_list([c for c in categorical_candidates if c in X_dev.columns])
        numeric_cols = unique_list([c for c in selected_features if c not in categorical_cols])

        feature_union = unique_list(numeric_cols + categorical_cols)
        X_dev = X_dev[feature_union].copy()
        X_blind_raw = X_blind_raw[feature_union].copy()

        X_train_raw, X_test_raw, y_train_raw, y_test_raw, idx_train, idx_test = train_test_split(
            X_dev,
            y_dev,
            X_dev.index,
            test_size=test_size,
            random_state=int(random_state),
        )

        depth_train = dev_df.loc[idx_train, depth_col]
        depth_test = dev_df.loc[idx_test, depth_col]

        X_train_raw = X_train_raw.loc[:, ~pd.Index(X_train_raw.columns).duplicated()].copy()
        X_test_raw = X_test_raw.loc[:, ~pd.Index(X_test_raw.columns).duplicated()].copy()

        cleaner = DataCleaner(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            missing_method=missing_method,
            outlier_method=outlier_method,
            z_thresh=z_thresh,
            iqr_factor=iqr_factor,
            iso_contamination=iso_contamination,
        )

        cleaner.fit(X_train_raw)

        X_train_clean = cleaner.transform(
            X_train_raw,
            apply_row_removal=outlier_method in ["iqr_remove", "zscore_remove", "isolation_forest"],
        )

        y_train_clean = y_train_raw.loc[X_train_clean.index]
        depth_train_clean = depth_train.loc[X_train_clean.index]

        X_test_clean = cleaner.transform(X_test_raw, apply_row_removal=False)
        y_test_clean = y_test_raw.loc[X_test_clean.index]
        depth_test_clean = depth_test.loc[X_test_clean.index]

        X_blind_clean = cleaner.transform(X_blind_raw, apply_row_removal=False)
        y_blind_clean = y_blind.loc[X_blind_clean.index]
        depth_blind_clean = depth_blind.loc[X_blind_clean.index]

        if len(X_train_clean) < 5:
            st.error("Too many rows were removed from training data by outlier method.")
            st.stop()

        st.subheader("Cleaning Report")
        report_df = pd.DataFrame([cleaner.report_])
        st.dataframe(report_df)

        results = []
        trained_objects = {}

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Metrics", "Plots", "Feature Importance", "SHAP", "Downloads"]
        )

        for model_name in model_names:
            model = get_model(model_name, rs=int(random_state))
            model.fit(X_train_clean, y_train_clean)

            y_pred_test = model.predict(X_test_clean)
            y_pred_blind = model.predict(X_blind_clean)

            test_metrics = regression_metrics(y_test_clean, y_pred_test)
            blind_metrics = regression_metrics(y_blind_clean, y_pred_blind)

            results.append(
                {
                    "Model": model_name,
                    "Test_R2": test_metrics["R2"],
                    "Test_RMSE": test_metrics["RMSE"],
                    "Test_MAE": test_metrics["MAE"],
                    "Blind_R2": blind_metrics["R2"],
                    "Blind_RMSE": blind_metrics["RMSE"],
                    "Blind_MAE": blind_metrics["MAE"],
                }
            )

            trained_objects[model_name] = {
                "model": model,
                "X_train": X_train_clean,
                "y_train": y_train_clean,
                "X_test": X_test_clean,
                "y_test": y_test_clean,
                "X_blind": X_blind_clean,
                "y_blind": y_blind_clean,
                "depth_test": depth_test_clean,
                "depth_blind": depth_blind_clean,
                "y_pred_test": y_pred_test,
                "y_pred_blind": y_pred_blind,
                "cleaner": cleaner,
                "features": feature_union,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            }

        results_df = pd.DataFrame(results)

        with tab1:
            st.subheader("Model Metrics")
            st.dataframe(results_df.style.format("{:.4f}"))

        with tab2:
            for model_name in model_names:
                obj = trained_objects[model_name]

                st.markdown(f"### {model_name}")

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                axes[0, 0].scatter(obj["y_test"], obj["y_pred_test"], alpha=0.6)
                min_v = min(obj["y_test"].min(), obj["y_pred_test"].min())
                max_v = max(obj["y_test"].max(), obj["y_pred_test"].max())
                axes[0, 0].plot([min_v, max_v], [min_v, max_v], "r--")
                axes[0, 0].set_title("Test: Actual vs Predicted")
                axes[0, 0].set_xlabel("Actual")
                axes[0, 0].set_ylabel("Predicted")
                axes[0, 0].grid(True, alpha=0.3)

                residuals = obj["y_test"] - obj["y_pred_test"]
                axes[0, 1].hist(residuals, bins=30, alpha=0.7)
                axes[0, 1].set_title("Test Residual Distribution")
                axes[0, 1].set_xlabel("Residual")
                axes[0, 1].grid(True, alpha=0.3)

                test_plot_df = pd.DataFrame(
                    {
                        depth_col: obj["depth_test"],
                        "Actual": obj["y_test"],
                        "Predicted": obj["y_pred_test"],
                    }
                )
                safe_depth_sort_plot(
                    axes[1, 0],
                    test_plot_df,
                    depth_col=depth_col,
                    actual_col="Actual",
                    pred_col="Predicted",
                    title="Test: ROP vs Depth",
                )

                blind_plot_df = pd.DataFrame(
                    {
                        depth_col: obj["depth_blind"],
                        "Actual": obj["y_blind"],
                        "Predicted": obj["y_pred_blind"],
                    }
                )
                safe_depth_sort_plot(
                    axes[1, 1],
                    blind_plot_df,
                    depth_col=depth_col,
                    actual_col="Actual",
                    pred_col="Predicted",
                    title="Blind: ROP vs Depth",
                )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with tab3:
            for model_name in model_names:
                obj = trained_objects[model_name]
                model = obj["model"]

                st.markdown(f"### {model_name}")

                if hasattr(model, "feature_importances_"):
                    imp_df = pd.DataFrame(
                        {
                            "Feature": obj["X_train"].columns,
                            "Importance": model.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(imp_df["Feature"], imp_df["Importance"])
                    ax.invert_yaxis()
                    ax.set_title(f"{model_name} Feature Importance")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(imp_df)

                elif hasattr(model, "coef_"):
                    coef_df = pd.DataFrame(
                        {
                            "Feature": obj["X_train"].columns,
                            "Coefficient": model.coef_,
                        }
                    ).sort_values("Coefficient", ascending=False)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(coef_df["Feature"], coef_df["Coefficient"])
                    ax.invert_yaxis()
                    ax.set_title(f"{model_name} Coefficients")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(coef_df)

                else:
                    st.info("No feature importance available for this model.")

        with tab4:
            for model_name in model_names:
                obj = trained_objects[model_name]
                model = obj["model"]

                st.markdown(f"### {model_name}")

                try:
                    sample_n = min(200, len(obj["X_test"]))
                    if len(obj["X_test"]) > sample_n:
                        X_shap = obj["X_test"].sample(sample_n, random_state=42)
                    else:
                        X_shap = obj["X_test"]

                    explainer = shap.Explainer(model, obj["X_train"])
                    shap_values = explainer(X_shap)

                    fig = plt.figure()
                    shap.summary_plot(shap_values, X_shap, show=False)
                    st.pyplot(fig, clear_figure=True)
                    plt.close()

                except Exception as e:
                    st.warning(f"SHAP could not be generated for {model_name}: {e}")

        with tab5:
            st.subheader("Downloads")

            for model_name in model_names:
                obj = trained_objects[model_name]

                package = {
                    "model": obj["model"],
                    "cleaner": obj["cleaner"],
                    "features": obj["features"],
                    "numeric_cols": obj["numeric_cols"],
                    "categorical_cols": obj["categorical_cols"],
                }

                model_buffer = io.BytesIO()
                joblib.dump(package, model_buffer)
                model_buffer.seek(0)

                st.download_button(
                    label=f"Download {model_name} model package (.pkl)",
                    data=model_buffer,
                    file_name=f"{model_name.replace(' ', '_').lower()}_rop_model.pkl",
                    mime="application/octet-stream",
                )

                pred_test_df = pd.DataFrame(
                    {
                        "Actual_Test": obj["y_test"].values,
                        "Predicted_Test": obj["y_pred_test"],
                    },
                    index=obj["y_test"].index,
                )

                pred_blind_df = pd.DataFrame(
                    {
                        "Actual_Blind": obj["y_blind"].values,
                        "Predicted_Blind": obj["y_pred_blind"],
                    },
                    index=obj["y_blind"].index,
                )

                pred_buffer_test = io.BytesIO()
                pred_test_df.to_csv(pred_buffer_test, index=True)
                pred_buffer_test.seek(0)

                st.download_button(
                    label=f"Download {model_name} test predictions (.csv)",
                    data=pred_buffer_test,
                    file_name=f"{model_name.replace(' ', '_').lower()}_test_predictions.csv",
                    mime="text/csv",
                )

                pred_buffer_blind = io.BytesIO()
                pred_blind_df.to_csv(pred_buffer_blind, index=True)
                pred_buffer_blind.seek(0)

                st.download_button(
                    label=f"Download {model_name} blind predictions (.csv)",
                    data=pred_buffer_blind,
                    file_name=f"{model_name.replace(' ', '_').lower()}_blind_predictions.csv",
                    mime="text/csv",
                )

else:
    st.info("Please upload a CSV or Excel file to begin.")
