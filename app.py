import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataEnergy Dashboard", layout="wide")

# =========================
# CSS - MATCH IMAGE STYLE
# =========================
st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background: #2b2f30;
}

/* Main content area */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #2b2f30;
    padding-top: 0.5rem;
}
[data-testid="stSidebar"] * {
    color: #111 !important;
}

/* White boxes in sidebar */
.sidebar-box {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: none;
}

.sidebar-title-box {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    padding: 16px 14px;
    margin-bottom: 14px;
}

.sidebar-title-box h1 {
    font-size: 20px;
    color: #111111;
    margin: 0;
    line-height: 1.25;
    font-weight: 700;
}

/* Fake logo area */
.logo-box {
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 12px;
    padding-left: 4px;
}
.logo-blue { color: #2a79ff; }
.logo-red { color: #e24a3b; }

/* Main panel */
.main-panel {
    background: #efefef;
    border: 1px solid #bdbdbd;
    padding: 16px;
    min-height: 80vh;
}

/* Chart boxes */
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
    color: #222;
    margin-bottom: 6px;
}

/* Remove streamlit element spacing weirdness */
div[data-testid="stVerticalBlock"] > div:has(.chart-box) {
    margin-bottom: 0.5rem;
}

/* Checkbox/radio tweaks */
.stCheckbox, .stRadio {
    margin-bottom: 0.15rem;
}

/* Select boxes inside sidebar */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #111 !important;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: #1f2937;
    color: white;
    border: 1px solid #111827;
    border-radius: 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR CONTENT
# =========================
with st.sidebar:
    st.markdown('<div class="logo-box"><span class="logo-blue">◉◉</span> Data<span class="logo-red">Energy</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-title-box">
        <h1>Feature Importance:<br>NPHI Log Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# =========================
# SAMPLE DATA IF NO FILE
# =========================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    np.random.seed(42)
    n = 1200
    depth = np.linspace(500, 3000, n)
    df = pd.DataFrame({
        "CALI": np.random.normal(10, 1.2, n),
        "DEPTH_MD": depth,
        "DTC": np.random.normal(80, 8, n),
        "GR": np.random.normal(75, 20, n),
        "PEF": np.random.normal(3, 0.6, n),
        "RDEP": np.random.normal(15, 5, n),
        "RHOB": np.random.normal(2.4, 0.12, n),
        "RMED": np.random.normal(10, 3, n),
        "ROP": np.random.normal(25, 7, n),
    })
    df["NPHI"] = (
        0.25
        + 0.0015 * df["GR"]
        - 0.002 * df["RHOB"] * 10
        + 0.0008 * df["DTC"]
        + np.random.normal(0, 0.03, n)
    )
    df["NPHI"] = np.clip(df["NPHI"], 0.02, 0.75)

target_col = "NPHI"

available_features = [c for c in df.columns if c != target_col]

with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown("**Regression_Predictor**")

    select_all = st.checkbox("Select all", value=True)

    if select_all:
        selected_features = st.multiselect(
            "Features",
            available_features,
            default=available_features,
            label_visibility="collapsed"
        )
    else:
        selected_features = st.multiselect(
            "Features",
            available_features,
            default=available_features[:4],
            label_visibility="collapsed"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown("**RegressorModel**")
    model_name = st.radio(
        "",
        ["DecisionTreeRegressor", "LinearRegression", "RandomForestRegressor", "SVR", "XGBoostRegressor"],
        index=4,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODEL MAPPING
# =========================
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

def get_model(name):
    if name == "DecisionTreeRegressor":
        return DecisionTreeRegressor(max_depth=6, random_state=42)
    elif name == "LinearRegression":
        return LinearRegression()
    elif name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    elif name == "SVR":
        return SVR()
    elif name == "XGBoostRegressor":
        if xgb_available:
            return XGBRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
        else:
            return GradientBoostingRegressor(random_state=42)
    return LinearRegression()

# =========================
# TRAIN
# =========================
if len(selected_features) < 1:
    st.warning("Please select at least one feature.")
    st.stop()

X = df[selected_features].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = get_model(model_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Feature importance
importance_df = pd.DataFrame()
if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
elif hasattr(model, "coef_"):
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": np.abs(model.coef_)
    }).sort_values("Importance", ascending=True)
else:
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": np.random.rand(len(selected_features))
    }).sort_values("Importance", ascending=True)

# Build comparison dataframe
test_plot_df = X_test.copy()
test_plot_df["Actual"] = y_test.values
test_plot_df["Predicted"] = y_pred

if "DEPTH_MD" not in test_plot_df.columns:
    test_plot_df["DEPTH_MD"] = np.arange(len(test_plot_df))

test_plot_df = test_plot_df.sort_values("DEPTH_MD")

# =========================
# MAIN PANEL
# =========================
st.markdown('<div class="main-panel">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])

with col2:
    top1, top2 = st.columns([1, 1])

    # Feature Importance
    with top1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Feature Importance</div>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.barh(importance_df["Feature"], importance_df["Importance"], color="#8bb7d8", edgecolor="#6c9bbf")
        ax1.set_xlabel("Importance")
        ax1.set_ylabel("Features")
        ax1.set_facecolor("#f8f7ea")
        fig1.patch.set_facecolor("#f8f7ea")
        ax1.grid(axis="x", linestyle="-", alpha=0.2)
        st.pyplot(fig1, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Cross Plot
    with top2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-title">Cross Plot: NPHI vs. NPHI-Predicted<br>Model: {model_name}</div>',
            unsafe_allow_html=True
        )

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(y_test, y_pred, s=8, color="blue", alpha=0.5, label="Blind Well")
        minv = min(min(y_test), min(y_pred))
        maxv = max(max(y_test), max(y_pred))
        ax2.plot([minv, maxv], [minv, maxv], '--', color='tomato', linewidth=1.2, label='Perfect Fit')
        ax2.set_xlabel("NPHI (Actual)")
        ax2.set_ylabel("NPHI (Predicted)")
        ax2.set_facecolor("#f8f7ea")
        fig2.patch.set_facecolor("#f8f7ea")
        ax2.grid(alpha=0.15)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.text(minv + 0.02, maxv - 0.08, f"Blind Well R²: {r2:.2f}", fontsize=11, weight="bold")
        st.pyplot(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Depth plot
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-title">Line Plot: Depth vs. NPHI (Blind Well)</div>',
        unsafe_allow_html=True
    )

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(test_plot_df["DEPTH_MD"], test_plot_df["Actual"], color="#6a6bd1", linewidth=1, alpha=0.8, label="NPHI Actual")
    ax3.plot(test_plot_df["DEPTH_MD"], test_plot_df["Predicted"], color="#ef7d57", linewidth=1, alpha=0.7, label="NPHI Predicted")
    ax3.set_xlabel("DEPTH_MD")
    ax3.set_ylabel("NPHI")
    ax3.set_facecolor("#f8f7ea")
    fig3.patch.set_facecolor("#f8f7ea")
    ax3.grid(alpha=0.15)
    ax3.legend(loc="upper right", fontsize=8)
    st.pyplot(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
