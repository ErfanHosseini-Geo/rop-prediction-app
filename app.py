import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ROP Feature Importance Dashboard",
    layout="wide"
)

# =========================
# CSS STYLING
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #2b2f30;
}
.main .block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background: #2b2f30;
}
.main-panel {
    background: #efefef;
    border: 1px solid #bdbdbd;
    padding: 18px;
    min-height: 85vh;
    border-radius: 8px;
}
.chart-box {
    background: #f8f7ea;
    border: 1px solid #d5d5c8;
    padding: 12px;
    margin-bottom: 14px;
    border-radius: 6px;
}
.chart-title {
    text-align: center;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #222;
}
.metric-box {
    background: #f8f7ea;
    border: 1px solid #d5d5c8;
    padding: 14px;
    text-align: center;
    border-radius: 6px;
}
.metric-label {
    font-size: 13px;
    color: #555;
}
.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #111;
}
.logo-box {
    color: white;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}
.sidebar-note {
    color: white;
    font-size: 14px;
    margin-bottom: 14px;
}
.small-note {
    font-size: 12px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def make_unique_columns(columns):
    seen = {}
    new_cols = []
    duplicates_found = False

    for i, col in enumerate(columns):
        col = str(col).strip()

        if col == "" or col.lower().startswith("unnamed") or col.lower() == "nan":
            col = f"Column_{i}"

        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
            duplicates_found = True
        else:
            seen[col] = 0

        new_cols.append(col)

    return new_cols, duplicates_found


def load_data(file):
    if file is None:
        np.random.seed(42)
        n = 1500
        depth = np.linspace(1000, 4500, n)

        df = pd.DataFrame({
            "Depth": depth,
            "WOB": np.random.normal(25, 5, n),
            "RPM": np.random.normal(120, 20, n),
            "Torque": np.random.normal(12, 2.5, n),
            "SPP": np.random.normal(2500, 300, n),
            "FlowRate": np.random.normal(650, 80, n),
            "MW": np.random.normal(10.5, 0.7, n),
            "ECD": np.random.normal(11.2, 0.8, n),
            "GR": np.random.normal(85, 20, n),
            "RHOB": np.random.normal(2.35, 0.1, n),
            "NPHI": np.random.normal(0.32, 0.08, n),
        })

        df["ROP"] = (
            0.9 * df["WOB"]
            + 0.12 * df["RPM"]
            - 0.006 * df["SPP"]
            + 0.018 * df["FlowRate"]
            - 1.8 * df["MW"]
            - 1.2 * df["ECD"]
            + np.random.normal(0, 3, n)
        )

        return df, False

    try:
        if file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    df.columns, duplicates_found = make_unique_columns(df.columns)

    return df, duplicates_found


def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def guess_target_column(columns):
    priority = ["ROP", "rop", "Rate of Penetration", "rate_of_penetration"]
    for p in priority:
        for c in columns:
            if c == p:
                return c
    for c in columns:
        if "rop" in str(c).lower():
            return c
    return columns[0] if columns else None


def guess_depth_column(columns):
    for c in columns:
        c_low = str(c).lower()
        if "depth" in c_low or c_low == "md" or "measured depth" in c_low:
            return c
    return columns[0] if columns else None


def get_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    elif name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    elif name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42)
    elif name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42, max_depth=8)
    elif name == "Linear Regression":
        return LinearRegression()
    else:
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


def compute_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if np.ndim(coef) > 1:
            coef = np.ravel(coef)
        importance = np.abs(coef)
    else:
        importance = np.zeros(len(feature_names))

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="logo-box">⛏️ ROP Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">Upload drilling data and evaluate feature importance for ROP prediction.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

# =========================
# LOAD DATA
# =========================
df, duplicates_found = load_data(uploaded_file)

if duplicates_found:
    st.warning("Duplicate column names were found in the uploaded file and automatically renamed.")

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Clean index just in case
df = df.reset_index(drop=True)

# Numeric columns
numeric_cols = get_numeric_columns(df)

if len(numeric_cols) < 2:
    st.error("Not enough numeric columns found in the dataset.")
    st.stop()

default_target = guess_target_column(numeric_cols)
default_depth = guess_depth_column(df.columns.tolist())

# =========================
# SIDEBAR OPTIONS
# =========================
with st.sidebar:
    target_col = st.selectbox(
        "Select Target Column",
        options=numeric_cols,
        index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
    )

    depth_col = st.selectbox(
        "Select Depth Column",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(default_depth) if default_depth in df.columns else 0
    )

    feature_options = [c for c in numeric_cols if c != target_col]

    default_features = []
    preferred = ["WOB", "RPM", "Torque", "SPP", "FlowRate", "MW", "ECD", "GR"]
    for p in preferred:
        if p in feature_options:
            default_features.append(p)

    if len(default_features) == 0:
        default_features = feature_options[:min(8, len(feature_options))]

    selected_features = st.multiselect(
        "Select Input Features",
        options=feature_options,
        default=default_features
    )

    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "Extra Trees", "Gradient Boosting", "Decision Tree", "Linear Regression"]
    )

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

# =========================
# VALIDATION
# =========================
if not selected_features:
    st.warning("Please select at least one input feature.")
    st.stop()

required_cols = list(dict.fromkeys([depth_col, target_col] + selected_features))

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"These required columns are missing from the dataset: {missing_cols}")
    st.stop()

data_model = df[required_cols].copy()

# Remove rows where target is missing
data_model = data_model.dropna(subset=[target_col])

if data_model.empty:
    st.error("No valid rows remain after removing missing target values.")
    st.stop()

X = data_model[selected_features].copy()
y = data_model[target_col].copy()

# Safety: make feature names unique again at X-level
if X.columns.duplicated().any():
    new_cols, _ = make_unique_columns(X.columns)
    X.columns = new_cols

# Impute missing feature values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

if len(X_imputed) < 5:
    st.error("Not enough valid records to train the model.")
    st.stop()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=test_size, random_state=42
)

# Model training
model = get_model(model_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Feature importance
importance_df = compute_feature_importance(model, X_train.columns.tolist())

# Prediction plot dataframe
plot_df = data_model.loc[X_test.index, [depth_col, target_col]].copy()
plot_df["Predicted_ROP"] = y_pred
plot_df = plot_df.sort_values(by=depth_col)

# Download results table
results_df = data_model.loc[X_test.index, [depth_col, target_col]].copy()
results_df["Predicted_ROP"] = y_pred

# =========================
# MAIN UI
# =========================
st.markdown('<div class="main-panel">', unsafe_allow_html=True)

st.title("ROP Feature Importance Analysis")
st.caption("Analyze drilling parameters affecting Rate of Penetration (ROP)")

cmeta1, cmeta2, cmeta3 = st.columns(3)
with cmeta1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Rows</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with cmeta2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Numeric Columns</div>
        <div class="metric-value">{len(numeric_cols)}</div>
    </div>
    """, unsafe_allow_html=True)

with cmeta3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Model</div>
        <div class="metric-value" style="font-size:18px;">{model_name}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{r2:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">{rmse:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">MAE</div>
        <div class="metric-value">{mae:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Feature Importance for ROP Prediction</div>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    imp_plot = importance_df.sort_values("Importance", ascending=True)
    ax1.barh(imp_plot["Feature"], imp_plot["Importance"], color="#8bb7d8", edgecolor="#6c9bbf")
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("Features")
    ax1.grid(axis="x", alpha=0.2)
    ax1.set_facecolor("#f8f7ea")
    fig1.patch.set_facecolor("#f8f7ea")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Actual vs Predicted ROP</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(y_test, y_pred, s=18, color="blue", alpha=0.5)
    minv = min(np.min(y_test), np.min(y_pred))
    maxv = max(np.max(y_test), np.max(y_pred))
    ax2.plot([minv, maxv], [minv, maxv], "--", color="tomato")
    ax2.set_xlabel("Actual ROP")
    ax2.set_ylabel("Predicted ROP")
    ax2.grid(alpha=0.2)
    ax2.set_facecolor("#f8f7ea")
    fig2.patch.set_facecolor("#f8f7ea")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">ROP vs Depth</div>', unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(plot_df[depth_col], plot_df[target_col], label="Actual ROP", color="#6a6bd1", linewidth=1.2)
ax3.plot(plot_df[depth_col], plot_df["Predicted_ROP"], label="Predicted ROP", color="#ef7d57", linewidth=1.2)
ax3.set_xlabel(depth_col)
ax3.set_ylabel("ROP")
ax3.legend()
ax3.grid(alpha=0.2)
ax3.set_facecolor("#f8f7ea")
fig3.patch.set_facecolor("#f8f7ea")
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

st.markdown('</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)

with t1:
    st.subheader("Feature Importance Table")
    st.dataframe(importance_df, use_container_width=True)

with t2:
    st.subheader("Prediction Sample")
    st.dataframe(results_df.head(50), use_container_width=True)

importance_csv = importance_df.to_csv(index=False).encode("utf-8")
results_csv = results_df.to_csv(index=False).encode("utf-8")

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Download Feature Importance CSV",
        data=importance_csv,
        file_name="rop_feature_importance.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        "Download Prediction Results CSV",
        data=results_csv,
        file_name="rop_predictions.csv",
        mime="text/csv"
    )

with st.expander("Preview Uploaded Data"):
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown(f'<div class="small-note">Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
