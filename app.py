import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ROP Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #f3f4f6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stButton button {
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 700;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
}

/* Main container */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* Header */
.dashboard-header {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
.dashboard-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.dashboard-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
}

/* Cards */
.card {
    background: white;
    border-radius: 18px;
    padding: 1rem 1rem 0.6rem 1rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    margin-bottom: 1rem;
    border: 1px solid #e5e7eb;
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.6rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    text-align: center;
}
.metric-label {
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}
.metric-number {
    color: #111827;
    font-size: 1.8rem;
    font-weight: 800;
}

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #111827;
    margin: 0.3rem 0 0.8rem 0.2rem;
}

/* Sidebar logo/title block */
.sidebar-brand {
    background: rgba(255,255,255,0.06);
    padding: 1rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.sidebar-brand h2 {
    color: white;
    margin: 0;
    font-size: 1.3rem;
}
.sidebar-brand p {
    color: #cbd5e1;
    margin: 0.3rem 0 0 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
