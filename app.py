import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.set_page_config(page_title="Income Classifier", layout="wide")

st.title("Adult Income Classification System")
st.markdown(
    "Predict whether an individual's income exceeds **$50K/year** using multiple machine learning models."
)
st.write("Upload test data and evaluate multiple ML models.")

st.sidebar.header("Controls")

MODEL_DIR = "models"

models = {
    file.replace(".pkl", ""): os.path.join(MODEL_DIR, file)
    for file in os.listdir(MODEL_DIR)
    if file.endswith(".pkl")
}

model_list = list(models.keys())
if model_list:
    model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
    )
else:
    st.error("No models found in the /models directory!")
    st.stop()

st.info(f"Currently using **{model_name.replace('_',' ').title()}**")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    if "income" in df.columns:
        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

        X = df.drop("income", axis=1)
        y = df["income"]

        model = joblib.load(models[model_name])

        with st.spinner("Running inference..."):
            preds = model.predict(X)

        st.subheader("Classification Report")
        report = classification_report(y, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, preds)
        cm_df = pd.DataFrame(
                cm,
                index=["Actual <=50K", "Actual >50K"],
                columns=["Predicted <=50K", "Predicted >50K"]
                )

        st.dataframe(cm_df)

    else:
        st.error("Dataset must contain the 'income' column.")
