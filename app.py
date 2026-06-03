import io
import joblib
import shap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Advanced ROP Dashboard", layout="wide")
st.title("Advanced ROP Prediction Dashboard")
st.markdown("Development Well → Train/Test | Blind Well → Final Unseen Evaluation")

# =========================================================
# SAFE COLUMN HANDLING
# =========================================================
def make_unique_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        col = str(col).strip()
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    return new_cols

def safe_boolean_mask(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = make_unique_columns(df.columns)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.reset_index(drop=True)

    return df

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload dataset to start.")
    st.stop()

df = load_data(uploaded_file)
df = df.replace(-999, np.nan)

st.subheader("Data Preview")
st.dataframe(df.head())

# =========================================================
# COLUMN SELECTION
# =========================================================
well_col = st.sidebar.selectbox("Well Column", df.columns)
target_col = st.sidebar.selectbox("Target Column (ROP)", df.columns)
depth_col = st.sidebar.selectbox("Depth Column", df.columns)

# Remove duplicates safely
all_cols = list(dict.fromkeys(df.columns))

candidate_features = [
    c for c in all_cols
    if c not in [well_col, target_col, depth_col]
]

selected_features = st.sidebar.multiselect(
    "Select Features",
    candidate_features,
    default=candidate_features[:5] if len(candidate_features) >= 5 else candidate_features
)

selected_features = list(dict.fromkeys(selected_features))

# =========================================================
# WELL SELECTION
# =========================================================
wells = sorted(df[well_col].astype(str).unique())

if len(wells) < 2:
    st.error("Need at least 2 wells.")
    st.stop()

dev_well = st.sidebar.selectbox("Development Well", wells)
blind_well = st.sidebar.selectbox("Blind Well", [w for w in wells if w != dev_well])

# =========================================================
# MODEL SELECTION
# =========================================================
model_names = st.sidebar.multiselect(
    "Models",
    ["Random Forest", "Extra Trees", "Gradient Boosting", "Linear Regression"],
    default=["Random Forest"]
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
random_state = st.sidebar.number_input("Random State", value=42)

if not st.sidebar.button("Run Modeling"):
    st.stop()

# =========================================================
# SAFE DATA PREPARATION
# =========================================================
final_cols = [well_col, target_col, depth_col] + selected_features
final_cols = list(dict.fromkeys(final_cols))

work_df = df.loc[:, final_cols].copy()
work_df = work_df.loc[:, ~work_df.columns.duplicated()]

well_series = safe_boolean_mask(work_df[well_col])

dev_mask = well_series.astype(str) == str(dev_well)
blind_mask = well_series.astype(str) == str(blind_well)

dev_df = work_df.loc[dev_mask].dropna(subset=[target_col])
blind_df = work_df.loc[blind_mask].dropna(subset=[target_col])

if dev_df.empty or blind_df.empty:
    st.error("One of the wells has no valid data.")
    st.stop()

X_dev = dev_df[selected_features]
y_dev = dev_df[target_col]

X_blind = blind_df[selected_features]
y_blind = blind_df[target_col]
depth_blind = blind_df[depth_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=test_size, random_state=random_state
)

# =========================================================
# PREPROCESSOR (NO LEAKAGE)
# =========================================================
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in selected_features if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]),
     categorical_cols)
])

preprocessor.fit(X_train)

X_train = pd.DataFrame(preprocessor.transform(X_train), columns=numeric_cols + categorical_cols)
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=numeric_cols + categorical_cols)
X_blind = pd.DataFrame(preprocessor.transform(X_blind), columns=numeric_cols + categorical_cols)

# =========================================================
# MODELING
# =========================================================
def get_model(name):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=random_state),
        "Linear Regression": LinearRegression(),
    }
    return models[name]

def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }

results = []
trained_models = {}

for name in model_names:
    model = get_model(name)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_blind = model.predict(X_blind)

    test_m = metrics(y_test, y_pred_test)
    blind_m = metrics(y_blind, y_pred_blind)

    results.append({
        "Model": name,
        "Test_R2": test_m["R2"],
        "Test_RMSE": test_m["RMSE"],
        "Test_MAE": test_m["MAE"],
        "Blind_R2": blind_m["R2"],
        "Blind_RMSE": blind_m["RMSE"],
        "Blind_MAE": blind_m["MAE"],
    })

    trained_models[name] = (model, y_pred_blind)

results_df = pd.DataFrame(results)

# ✅ Safe Display (NO Styler)
num_cols = results_df.select_dtypes(include=[np.number]).columns
results_df[num_cols] = results_df[num_cols].round(4)

st.subheader("Model Performance")
st.dataframe(results_df)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
for name in model_names:
    model, _ = trained_models[name]
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
# SHAP (SAFE)
# =========================================================
for name in model_names:
    model, _ = trained_models[name]
    st.subheader(f"{name} SHAP Summary")

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        fig = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
        plt.close()
    except Exception:
        st.info("SHAP not supported for this model.")

# =========================================================
# DOWNLOAD MODEL
# =========================================================
for name in model_names:
    model, _ = trained_models[name]

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    st.download_button(
        label=f"Download {name} Model (.pkl)",
        data=buffer,
        file_name=f"{name.replace(' ','_').lower()}_model.pkl"
    )
