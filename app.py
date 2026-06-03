حتماً. این **نسخه نهایی کامل و یکپارچه `app.py`** است که:

- **Missing value imputation** با روش‌های مختلف:
  - `median`
  - `mean`
  - `most_frequent`
  - `knn`
- **Outlier handling** با روش‌های مختلف:
  - `none`
  - `iqr_capping`
  - `iqr_remove`
  - `zscore_remove`
  - `isolation_forest`
- جلوگیری از **data leakage**
- clean شدن داده‌ها قبل از مدل‌سازی
- ارزیابی روی **test** و **blind**
- نمودارها، SHAP، feature importance
- دانلود مدل و خروجی پیش‌بینی

فقط کل این فایل را جایگزین `app.py` کن:

```python
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


# =================================
# Page Config
# =================================
st.set_page_config(page_title="Advanced ROP Dashboard", layout="wide")
st.title("🚀 Advanced ROP Prediction Dashboard")
st.markdown("Development Well → Train/Test | Blind Well → Final Unseen Evaluation")


# =================================
# Helpers
# =================================
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
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = make_unique_columns(df.columns)
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


# =================================
# Custom Data Cleaner
# =================================
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
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
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

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_imputer, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ],
            remainder="drop",
        )

    def fit(self, X):
        X = X.copy()
        self._build_preprocessor()

        missing_before = int(X.isna().sum().sum())

        X_imp = self.preprocessor.fit_transform(X)
        all_cols = self.numeric_cols + self.categorical_cols
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
                q1 = X_imp_df[col].quantile(
