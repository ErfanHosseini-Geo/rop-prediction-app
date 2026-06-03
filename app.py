import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


st.title("ROP Prediction App")


uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx","csv"])


def load_data(file):

    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    return df


if uploaded_file:

    df = load_data(uploaded_file)

    st.write("Data Preview")
    st.dataframe(df.head())

    # تبدیل -999 به NaN
    df = df.replace(-999, np.nan)

    if "ROP" not in df.columns:
        st.error("ROP column not found")
        st.stop()


    # حذف ستون های غیرعددی
    df_numeric = df.select_dtypes(include=np.number)

    # حذف ستون target از feature
    X = df_numeric.drop(columns=["ROP"])
    y = df_numeric["ROP"]

    # حذف ROPA برای جلوگیری از leakage
    if "ROPA" in X.columns:
        X = X.drop(columns=["ROPA"])


    st.write("Features used:", list(X.columns))


    test_size = st.slider("Test Size",0.1,0.4,0.2)

    model_name = st.selectbox(
        "Select Model",
        ["Random Forest","Extra Trees"]
    )


    if st.button("Run Model"):

        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=200)

        if model_name == "Extra Trees":
            model = ExtraTreesRegressor(n_estimators=200)


        pipeline = Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",model)
        ])


        X_train,X_test,y_train,y_test = train_test_split(
            X,y,
            test_size=test_size,
            random_state=42
        )


        pipeline.fit(X_train,y_train)

        y_pred = pipeline.predict(X_test)


        r2 = r2_score(y_test,y_pred)
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        mae = mean_absolute_error(y_test,y_pred)


        st.subheader("Model Performance")

        col1,col2,col3 = st.columns(3)

        col1.metric("R2",round(r2,3))
        col2.metric("RMSE",round(rmse,3))
        col3.metric("MAE",round(mae,3))


        # Actual vs predicted

        fig,ax = plt.subplots()

        ax.scatter(y_test,y_pred)

        ax.set_xlabel("Actual ROP")
        ax.set_ylabel("Predicted ROP")

        st.pyplot(fig)


        # feature importance

        model = pipeline.named_steps["model"]

        importances = model.feature_importances_

        imp = pd.DataFrame({
            "feature":X.columns,
            "importance":importances
        }).sort_values("importance",ascending=False)


        st.subheader("Feature Importance")

        fig2,ax2 = plt.subplots()

        ax2.barh(imp["feature"],imp["importance"])

        ax2.invert_yaxis()

        st.pyplot(fig2)


        # prediction table

        results = X_test.copy()

        results["Actual_ROP"] = y_test
        results["Predicted_ROP"] = y_pred

        st.subheader("Prediction Results")

        st.dataframe(results.head(20))
