import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ROP Feature Importance Dashboard", layout="wide")

# =========================
# CSS
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
}
.chart-box {
    background: #f8f7ea;
    border: 1px solid #d5d5c8;
    padding: 10px;
    margin-bottom: 14px;
}
.chart-title {
    text-align: center;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #222;
}
.metric-box {
    background: #f8f7ea;
    border: 1px solid #d5d5c8;
    padding: 12px;
    text-align: center;
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
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 12px;
}
.sidebar-note {
    color: white;
    font-size: 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="logo-box">⛏️ ROP Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Upload drilling data and evaluate feature importance for ROP prediction.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

# =========================
# LOAD DATA
# =========================
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

        return df

    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

df = load_data(uploaded_file)

# =========================
# DATA PREP
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Not enough numeric columns found in the dataset.")
    st.stop()

with st.sidebar:
    target_col = st.selectbox(
        "Select Target Column",
        options=numeric_cols,
        index=numeric_cols.index("ROP") if "ROP" in numeric_cols else 0
    )

    depth_candidates = [c for c in df.columns if "depth" in c.lower() or "md" in c.lower()]
    default_depth = depth_candidates[0] if depth_candidates else df.columns[0]

    depth_col = st.selectbox(
        "Select Depth Column",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(default_depth)
    )

    feature_options = [c for c in numeric_cols if c != target_col]

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
# MODEL SELECTION
# =========================
def get_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    elif name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=300, random_state=42)
    elif name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42)
    elif name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42, max_depth=8)
    elif name == "Linear Regression":
        return LinearRegression()
    return RandomForestRegressor(n_estimators=300, random_state=42)

if len(selected_features) < 1:
    st.warning("Please select at least one feature.")
    st.stop()

data_model = df[[depth_col, target_col] + selected_features].copy()
data_model = data_model.dropna(axis=0)

X = data_model[selected_features]
y = data_model[target_col]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=test_size, random_state=42
)

model = get_model(model_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# =========================
# FEATURE IMPORTANCE
# =========================
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
elif hasattr(model, "coef_"):
    importance = np.abs(model.coef_)
else:
    importance = np.zeros(len(selected_features))

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": importance
}).sort_values("Importance", ascending=True)

# =========================
# DEPTH TREND DATA
# =========================
plot_df = data_model.loc[X_test.index, [depth_col, target_col]].copy()
plot_df["Predicted_ROP"] = y_pred
plot_df = plot_df.sort_values(depth_col)

# =========================
# MAIN PANEL
# =========================
st.markdown('<div class="main-panel">', unsafe_allow_html=True)

st.title("ROP Feature Importance Analysis")
st.caption("Analyze drilling parameters affecting Rate of Penetration (ROP)")

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

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Feature Importance for ROP Prediction</div>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.barh(importance_df["Feature"], importance_df["Importance"], color="#8bb7d8", edgecolor="#6c9bbf")
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("Features")
    ax1.grid(axis="x", alpha=0.2)
    ax1.set_facecolor("#f8f7ea")
    fig1.patch.set_facecolor("#f8f7ea")
    st.pyplot(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Actual vs Predicted ROP</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(y_test, y_pred, s=15, color="blue", alpha=0.5)
    minv = min(min(y_test), min(y_pred))
    maxv = max(max(y_test), max(y_pred))
    ax2.plot([minv, maxv], [minv, maxv], "--", color="tomato")
    ax2.set_xlabel("Actual ROP")
    ax2.set_ylabel("Predicted ROP")
    ax2.grid(alpha=0.2)
    ax2.set_facecolor("#f8f7ea")
    fig2.patch.set_facecolor("#f8f7ea")
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">ROP vs Depth</div>', unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(plot_df[depth_col], plot_df[target_col], label="Actual ROP", color="#6a6bd1", linewidth=1)
ax3.plot(plot_df[depth_col], plot_df["Predicted_ROP"], label="Predicted ROP", color="#ef7d57", linewidth=1)
ax3.set_xlabel(depth_col)
ax3.set_ylabel("ROP")
ax3.legend()
ax3.grid(alpha=0.2)
ax3.set_facecolor("#f8f7ea")
fig3.patch.set_facecolor("#f8f7ea")
st.pyplot(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Feature Importance Table")
st.dataframe(importance_df.sort_values("Importance", ascending=False), use_container_width=True)

csv = importance_df.sort_values("Importance", ascending=False).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Feature Importance CSV",
    data=csv,
    file_name="rop_feature_importance.csv",
    mime="text/csv"
)

st.markdown('</div>', unsafe_allow_html=True)
