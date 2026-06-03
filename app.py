import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Advanced ROP Analytics", layout="wide")
st.title("🚀 Advanced ROP Prediction Dashboard")

# ---------------------------------
# Helpers
# ---------------------------------
def make_unique_columns(columns):
    seen = {}
    new_cols = []
    for i, col in enumerate(columns):
        col = str(col).strip() if col else f"Unnamed_{i}"
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    return new_cols

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = make_unique_columns(df.columns)
    return df

def get_model(name, rs):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=rs),
        "Linear Regression": LinearRegression()
    }
    return models[name]

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("📂 Data & Model Setup")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if not uploaded_file:
    st.info("Waiting for data upload...")
    st.stop()

df = load_data(uploaded_file).replace(-999, np.nan)
st.sidebar.success("Data Loaded!")

# Well Selection
wells = sorted(df['WELL'].unique().tolist())
dev_well = st.sidebar.selectbox("Development Well (Train/Test)", wells, index=0)
blind_well = st.sidebar.selectbox("Blind Well (Unseen)", [w for w in wells if w != dev_well])

# Model Selection (Multiple)
selected_model_names = st.sidebar.multiselect("Models to Compare", 
                                             ["Random Forest", "Extra Trees", "Gradient Boosting", "Linear Regression"],
                                             default=["Extra Trees"])

# Features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in ["FORMATION", "GROUP"] if c in df.columns]
selected_features = st.sidebar.multiselect("Input Features", numeric_cols + cat_cols, 
                                          default=[c for c in numeric_cols if c not in ["ROP", "ROPA"]])

if not selected_model_names or not selected_features:
    st.warning("Select at least one model and one feature.")
    st.stop()

run_btn = st.sidebar.button("🚀 Train & Evaluate")

if run_btn:
    # -----------------------------
    # Processing
    # -----------------------------
    data = df.copy()
    
    # Label Encoding for Categorical
    for col in cat_cols:
        if col in selected_features:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Split Data
    dev_df = data[data['WELL'] == dev_well].dropna(subset=['ROP'])
    blind_df = data[data['WELL'] == blind_well].dropna(subset=['ROP'])
    
    X_dev = dev_df[selected_features]
    y_dev = dev_df['ROP']
    X_blind = blind_df[selected_features]
    y_blind = blind_df['ROP']

    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

    results = {}
    trained_pipelines = {}

    for name in selected_model_names:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", get_model(name, 42))
        ])
        pipe.fit(X_train, y_train)
        
        y_test_pred = pipe.predict(X_test)
        y_blind_pred = pipe.predict(X_blind)
        
        results[name] = {
            "test_metrics": [r2_score(y_test, y_test_pred), np.sqrt(mean_squared_error(y_test, y_test_pred))],
            "blind_metrics": [r2_score(y_blind, y_blind_pred), np.sqrt(mean_squared_error(y_blind, y_blind_pred))],
            "blind_df": pd.DataFrame({"Actual": y_blind, "Pred": y_blind_pred, "Depth": blind_df["DEPTH_MD"] if "DEPTH_MD" in blind_df else blind_df.index})
        }
        trained_pipelines[name] = pipe

    # -----------------------------
    # Tabs
    # -----------------------------
    t1, t2, t3, t4, t5 = st.tabs(["📊 Performance", "📈 Well Logs", "🧠 Interpretability (SHAP)", "💾 Export", "📄 Data Preview"])

    with t1:
        st.subheader("Model Comparison")
        metrics_df = pd.DataFrame({
            name: [res["test_metrics"][0], res["test_metrics"][1], res["blind_metrics"][0], res["blind_metrics"][1]]
            for name, res in results.items()
        }, index=["Test R2", "Test RMSE", "Blind R2", "Blind RMSE"]).T
        st.table(metrics_df.style.highlight_max(axis=0, subset=["Test R2", "Blind R2"], color='lightgreen'))

    with t2:
        st.subheader("ROP vs Depth (Blind Well)")
        for name in selected_model_names:
            res_df = results[name]["blind_df"].sort_values("Depth")
            fig, ax = plt.subplots(figsize=(5, 8))
            ax.plot(res_df["Actual"], res_df["Depth"], label="Actual", color='black', alpha=0.5)
            # Smooth curve using rolling mean
            ax.plot(res_df["Pred"].rolling(window=5).mean(), res_df["Depth"], label=f"Pred ({name})", linewidth=2)
            ax.invert_yaxis()
            ax.set_title(f"Well: {blind_well} | Model: {name}")
            ax.legend()
            st.pyplot(fig)

    with t3:
        st.subheader("SHAP Feature Importance (Last Model)")
        last_name = selected_model_names[-1]
        model_obj = trained_pipelines[last_name].named_steps["model"]
        # Use a sample for SHAP to speed up
        explainer = shap.Explainer(model_obj)
        X_sample = Pipeline(trained_pipelines[last_name].steps[:-1]).transform(X_test[:100])
        shap_values = explainer(X_sample)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, feature_names=selected_features, show=False)
        st.pyplot(plt.gcf())

    with t4:
        st.subheader("Save & Download")
        for name in selected_model_names:
            buffer = io.BytesIO()
            joblib.dump(trained_pipelines[name], buffer)
            st.download_button(f"Download {name} Model (.pkl)", data=buffer.getvalue(), file_name=f"{name}_rop_model.pkl")

    with t5:
        st.subheader("Raw Data Preview")
        st.write(df.head(100))
