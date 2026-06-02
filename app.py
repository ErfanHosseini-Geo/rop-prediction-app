import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="ROP Prediction", layout="wide")
st.title("🛢️ ROP Prediction App")
st.markdown("آپلود فایل اکسل داده‌های حفاری، انتخاب ROP و آموزش مدل‌های مختلف")

st.sidebar.header("⚙️ تنظیمات")
uploaded_file = st.sidebar.file_uploader("فایل اکسل را آپلود کن", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("از سایدبار فایل اکسل را آپلود کن.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"خطا در خواندن فایل: {e}")
    st.stop()

st.subheader("نمایش داده")
st.dataframe(df.head())

st.subheader("اطلاعات کلی")
c1, c2, c3 = st.columns(3)
c1.metric("تعداد ردیف", df.shape[0])
c2.metric("تعداد ستون", df.shape[1])
c3.metric("تعداد خانه‌های خالی", int(df.isna().sum().sum()))

all_cols = df.columns.tolist()
target = st.selectbox("ستون هدف (ROP) را انتخاب کن", all_cols)

default_features = [c for c in all_cols if c != target]
features = st.multiselect("ویژگی‌ها را انتخاب کن", default_features, default=default_features)

if len(features) == 0:
    st.warning("حداقل یک ویژگی انتخاب کن.")
    st.stop()

X_full = df[features].copy()
num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_full.columns if c not in num_cols]

st.sidebar.subheader("پیش‌پردازش")
drop_target_na = st.sidebar.checkbox("حذف ردیف‌هایی که ROP خالی است", value=True)
scale_numeric = st.sidebar.checkbox("نرمال‌سازی ویژگی‌های عددی", value=True)

test_size = st.sidebar.slider("درصد داده تست", 10, 40, 20, 5)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

models = ["Linear Regression", "Random Forest", "SVR"]
if HAS_XGB:
    models.append("XGBoost")

model_name = st.sidebar.selectbox("مدل را انتخاب کن", models)

st.sidebar.subheader("تنظیمات مدل")
rf_n_estimators = st.sidebar.slider("RF: تعداد درخت", 50, 500, 200, 50)
rf_max_depth = st.sidebar.slider("RF: عمق بیشینه", 2, 30, 10, 1)

svr_C = st.sidebar.slider("SVR: C", 1, 100, 10, 1)
svr_epsilon = st.sidebar.slider("SVR: epsilon", 1, 100, 10, 1) / 100.0
svr_kernel = st.sidebar.selectbox("SVR: kernel", ["rbf", "linear", "poly"])

xgb_n_estimators = st.sidebar.slider("XGB: تعداد درخت", 50, 500, 200, 50) if HAS_XGB else 200
xgb_max_depth = st.sidebar.slider("XGB: عمق بیشینه", 2, 15, 6, 1) if HAS_XGB else 6
xgb_lr = st.sidebar.slider("XGB: learning_rate", 1, 50, 10, 1) / 100.0 if HAS_XGB else 0.1

do_cv = st.sidebar.checkbox("اجرای Cross Validation", value=False)
cv_folds = st.sidebar.slider("تعداد Fold", 3, 10, 5, 1)

data = df[features + [target]].copy()
if drop_target_na:
    data = data.dropna(subset=[target])

X = data[features].copy()
y = data[target].copy()

numeric_transformer_steps = [("imputer", SimpleImputer(strategy="median"))]
if scale_numeric:
    numeric_transformer_steps.append(("scaler", StandardScaler()))
numeric_transformer = Pipeline(steps=numeric_transformer_steps)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, [c for c in num_cols if c in X.columns]),
        ("cat", categorical_transformer, [c for c in cat_cols if c in X.columns]),
    ]
)

if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=random_state,
        n_jobs=-1
    )
elif model_name == "SVR":
    model = SVR(C=svr_C, epsilon=svr_epsilon, kernel=svr_kernel)
elif model_name == "XGBoost" and HAS_XGB:
    model = XGBRegressor(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_lr,
        random_state=random_state,
        n_jobs=-1
    )

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

if st.button("🚀 آموزش مدل و پیش‌بینی ROP"):
    if len(data) < 10:
        st.warning("داده خیلی کم است. حداقل 10 ردیف پیشنهاد می‌شود.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("📊 عملکرد مدل")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.4f}")
    m2.metric("RMSE", f"{rmse:.4f}")
    m3.metric("MAE", f"{mae:.4f}")

    if do_cv:
        st.subheader("Cross Validation")
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        st.write(f"میانگین R² در {cv_folds} فولد: {scores.mean():.4f}")
        st.write(f"امتیاز هر فولد: {np.round(scores, 4)}")

    results = pd.DataFrame({
        "Actual_ROP": y_test.values,
        "Predicted_ROP": y_pred
    })

    st.subheader("نمونه خروجی پیش‌بینی")
    st.dataframe(results.head(20))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y_test, y_pred, alpha=0.7)
        min_v = min(np.min(y_test), np.min(y_pred))
        max_v = max(np.max(y_test), np.max(y_pred))
        ax1.plot([min_v, max_v], [min_v, max_v], 'r--')
        ax1.set_xlabel("Actual ROP")
        ax1.set_ylabel("Predicted ROP")
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)

    with col2:
        st.subheader("Residual Plot")
        residuals = y_test.values - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_pred, residuals, alpha=0.7)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Predicted ROP")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    num_df = data.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("برای رسم Heatmap حداقل دو ستون عددی لازم است.")

    if model_name in ["Random Forest", "XGBoost"]:
        try:
            fitted_model = pipe.named_steps["model"]
            transformed_feature_names = []

            numeric_features_in_use = [c for c in num_cols if c in X.columns]
            transformed_feature_names.extend(numeric_features_in_use)

            categorical_features_in_use = [c for c in cat_cols if c in X.columns]
            if len(categorical_features_in_use) > 0:
                ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
                ohe_names = ohe.get_feature_names_out(categorical_features_in_use).tolist()
                transformed_feature_names.extend(ohe_names)

            importances = fitted_model.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": transformed_feature_names[:len(importances)],
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            st.subheader("Feature Importance")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=imp_df.head(20), x="Importance", y="Feature", ax=ax4)
            ax4.set_title("Top 20 Feature Importances")
            st.pyplot(fig4)
            st.dataframe(imp_df.head(20))
        except Exception as e:
            st.warning(f"نمایش Feature Importance ممکن نشد: {e}")

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ دانلود پیش‌بینی‌ها",
        data=csv,
        file_name="rop_predictions.csv",
        mime="text/csv"
    )
