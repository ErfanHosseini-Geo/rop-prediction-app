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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(page_title="Advanced ROP Dashboard", layout="wide")
st.title("Advanced ROP Prediction Dashboard")

# =========================================================
# Utility Functions
# =========================================================

def make_unique_columns(columns):
    seen = {}
    new_cols = []
    for i, col in enumerate(columns):
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


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def get_model(name, rs):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=rs, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=rs),
        "Linear Regression": LinearRegression(),
    }
    return models[name]

# =========================================================
# Data Cleaner (No Leakage)
# =========================================================

class DataCleaner:
    def __init__(self, numeric_cols, categorical_cols, missing_method="median"):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.missing_method = missing_method
        self.preprocessor = None

    def fit(self, X):
        if self.missing_method == "median":
            num_imputer = SimpleImputer(strategy="median")
        elif self.missing_method == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        else:
            num_imputer = KNNImputer(n_neighbors=5)

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        transformers = []
        if len(self.numeric_cols) > 0:
            transformers.append(("num", num_imputer, self.numeric_cols))
        if len(self.categorical_cols) > 0:
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(X)

    def transform(self, X):
        X_t = self.preprocessor.transform(X)
        cols = self.numeric_cols + self.categorical_cols
        return pd.DataFrame(X_t, columns=cols, index=X.index)

# =========================================================
# App Start
# =========================================================

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload dataset to start.")
    st.stop()

df = load_data(uploaded_file)
df = df.replace(-999, np.nan)

st.subheader("Preview")
st.dataframe(df.head())

well_col = st.sidebar.selectbox("Well Column", df.columns)
target_col = st.sidebar.selectbox("Target Column (ROP)", df.columns)
depth_col = st.sidebar.selectbox("Depth Column", df.columns)

wells = sorted(df[well_col].astype(str).unique())
dev_well = st.sidebar.selectbox("Development Well", wells)
blind_well = st.sidebar.selectbox("Blind Well", [w for w in wells if w != dev_well])

candidate_features = [c for c in df.columns if c not in [well_col, target_col]]
selected_features = st.sidebar.multiselect("Features", candidate_features, default=candidate_features[:5])

model_names = st.sidebar.multiselect(
    "Models",
    ["Random Forest", "Extra Trees", "Gradient Boosting", "Linear Regression"],
    default=["Random Forest"],
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
random_state = st.sidebar.number_input("Random State", value=42)

if not st.sidebar.button("Run Modeling"):
    st.stop()

# =========================================================
# Prepare Data
# =========================================================

selected_features = unique_list(selected_features)
work_df = df[[well_col, target_col, depth_col] + selected_features].copy()

dev_df = work_df[work_df[well_col] == dev_well].dropna(subset=[target_col])
blind_df = work_df[work_df[well_col] == blind_well].dropna(subset=[target_col])

X_dev = dev_df[selected_features]
y_dev = dev_df[target_col]

X_blind = blind_df[selected_features]
y_blind = blind_df[target_col]
depth_blind = blind_df[depth_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=test_size, random_state=random_state
)

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in selected_features if c not in categorical_cols]

cleaner = DataCleaner(numeric_cols, categorical_cols)
cleaner.fit(X_train)

X_train = cleaner.transform(X_train)
X_test = cleaner.transform(X_test)
X_blind = cleaner.transform(X_blind)

# =========================================================
# Modeling
# =========================================================

results = []
trained_models = {}

for name in model_names:
    model = get_model(name, random_state)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_blind = model.predict(X_blind)

    test_metrics = regression_metrics(y_test, y_pred_test)
    blind_metrics = regression_metrics(y_blind, y_pred_blind)

    results.append({
        "Model": name,
        "Test_R2": test_metrics["R2"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_MAE": test_metrics["MAE"],
        "Blind_R2": blind_metrics["R2"],
        "Blind_RMSE": blind_metrics["RMSE"],
        "Blind_MAE": blind_metrics["MAE"],
    })

    trained_models[name] = (model, y_pred_test, y_pred_blind)

results_df = pd.DataFrame(results)

# ✅ Safe Display (No Styler Crash)
st.subheader("Model Metrics")
numeric_cols_display = results_df.select_dtypes(include=[np.number]).columns
results_df[numeric_cols_display] = results_df[numeric_cols_display].round(4)
st.dataframe(results_df)

# =========================================================
# Feature Importance
# =========================================================

for name in model_names:
    model, _, _ = trained_models[name]
    st.subheader(f"{name} Feature Importance")

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(imp["Feature"], imp["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

    else:
        st.info("No feature importance available.")

# =========================================================
# SHAP
# =========================================================

for name in model_names:
    model, _, _ = trained_models[name]
    st.subheader(f"{name} SHAP Summary")

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        fig = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP not available: {e}")

# =========================================================
# Downloads
# =========================================================

for name in model_names:
    model, y_pred_test, y_pred_blind = trained_models[name]

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    st.download_button(
        label=f"Download {name} Model (.pkl)",
        data=buffer,
        file_name=f"{name.replace(' ','_').lower()}_model.pkl",
    )
