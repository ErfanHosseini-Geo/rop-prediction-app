import io
import joblib
import shap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import randint

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="ROP Journal Research Dashboard", layout="wide")
st.title("Journal-Level ROP Prediction Dashboard")

# =========================================================
# LOAD & CLEAN
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reset_index(drop=True)
    return df

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv","xlsx"])

if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)
df = df.replace(-999, np.nan)

# =========================================================
# COLUMN SELECTION
# =========================================================
well_col = st.sidebar.selectbox("Well Column", df.columns)
target_col = st.sidebar.selectbox("Target Column (ROP)", df.columns)
depth_col = st.sidebar.selectbox("Depth Column", df.columns)

candidate_features = [c for c in df.columns if c not in [well_col,target_col,depth_col]]
selected_features = st.sidebar.multiselect("Features", candidate_features, default=candidate_features[:6])

wells = sorted(df[well_col].astype(str).unique())
dev_well = st.sidebar.selectbox("Development Well", wells)
blind_well = st.sidebar.selectbox("Blind Well", [w for w in wells if w != dev_well])

# =========================================================
# DATA SPLIT
# =========================================================
dev_df = df[df[well_col].astype(str)==str(dev_well)].dropna(subset=[target_col])
blind_df = df[df[well_col].astype(str)==str(blind_well)].dropna(subset=[target_col])

X_dev = dev_df[selected_features]
y_dev = dev_df[target_col]

X_blind = blind_df[selected_features]
y_blind = blind_df[target_col]
depth_blind = blind_df[depth_col]

X_train, X_test, y_train, y_test = train_test_split(X_dev,y_dev,test_size=0.2,random_state=42)

# =========================================================
# PREPROCESSOR
# =========================================================
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in selected_features if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat",
     Pipeline([
         ("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))
     ]),
     categorical_cols)
])

# =========================================================
# MODELS & TUNING
# =========================================================
models = {
    "Random Forest": (
        RandomForestRegressor(random_state=42),
        {
            "model__n_estimators": randint(100,400),
            "model__max_depth": randint(5,30)
        }
    ),
    "Extra Trees": (
        ExtraTreesRegressor(random_state=42),
        {
            "model__n_estimators": randint(100,400),
            "model__max_depth": randint(5,30)
        }
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators": randint(100,400),
            "model__max_depth": randint(2,6)
        }
    ),
    "Linear Regression": (
        LinearRegression(),
        {}
    )
}

results = []

for name,(model,param_dist) in models.items():

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    if param_dist:
        search = RandomizedSearchCV(pipe,param_dist,n_iter=15,cv=5,n_jobs=-1,random_state=42)
        search.fit(X_train,y_train)
        best_model = search.best_estimator_
    else:
        best_model = pipe.fit(X_train,y_train)

    # Cross-validation
    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    cv_scores = []

    for train_idx,val_idx in kf.split(X_train):
        best_model.fit(X_train.iloc[train_idx],y_train.iloc[train_idx])
        pred = best_model.predict(X_train.iloc[val_idx])
        cv_scores.append(r2_score(y_train.iloc[val_idx],pred))

    cv_mean = np.mean(cv_scores)

    # Test
    y_pred_test = best_model.predict(X_test)
    y_pred_blind = best_model.predict(X_blind)

    test_r2 = r2_score(y_test,y_pred_test)
    blind_r2 = r2_score(y_blind,y_pred_blind)

    # Ranking score
    ranking_score = 0.5*cv_mean + 0.5*blind_r2

    results.append({
        "Model":name,
        "CV_R2":cv_mean,
        "Test_R2":test_r2,
        "Blind_R2":blind_r2,
        "Ranking_Score":ranking_score
    })

    # Residual Plot
    residuals = y_test - y_pred_test
    fig,ax = plt.subplots()
    ax.scatter(y_pred_test,residuals,alpha=0.5)
    ax.axhline(0,color='red')
    ax.set_xlabel("Predicted ROP")
    ax.set_ylabel("Residual")
    ax.set_title(f"{name} Residual Diagnostics")
    st.pyplot(fig)

    # Blind Well Depth Plot
    fig2,ax2 = plt.subplots()
    ax2.plot(y_blind,depth_blind,label="Actual")
    ax2.plot(y_pred_blind,depth_blind,label="Predicted")
    ax2.invert_yaxis()
    ax2.set_xlabel("ROP")
    ax2.set_ylabel("Depth")
    ax2.set_title(f"{name} Blind Well Performance")
    ax2.legend()
    st.pyplot(fig2)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Ranking_Score",ascending=False)
results_df = results_df.round(4)

st.subheader("Model Comparison & Ranking")
st.dataframe(results_df)

# =========================================================
# PUBLICATION-READY STYLE
# =========================================================
plt.style.use("seaborn-v0_8-paper")
fig,ax = plt.subplots(figsize=(6,4))
ax.bar(results_df["Model"],results_df["Ranking_Score"])
ax.set_ylabel("Ranking Score")
ax.set_title("Model Ranking Comparison")
plt.xticks(rotation=45)
st.pyplot(fig)
