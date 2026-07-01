import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from dotenv import load_dotenv
from functions.add_features import add_features
import re

st.set_page_config(
    page_title="Fake news detection",
    layout="wide"
)

st.title("Fake News Detection using Machine Learning and Deep Learning.")

@st.cache_resource
def load_model():
    load_dotenv()
    joblib_dir = os.getenv("JOBLIBPATH")

    file_path = os.path.join(joblib_dir, "XGBoost_pipeline_num.joblib")

    return joblib.load(file_path)



model = load_model()
radio = st.sidebar.radio("Navigation", ['Home', "Predict News", "Model Comparison", "About Project"])
if radio == "Home":
    st.markdown("""
            ## Project Overview

            This machine learning project predicts whether a message is fake or real
            based on its text and label.

            Throughout project developed, four models were trained including pre trained BERT model on over 40000 texts and achieved:

            - ROC-AUC: 99%
            - Accuracy: 99%
            - F1-score: 99.9%

            The application allows users to:

            - Predict probability of text being fake
            - Review model performance
            """)
if radio == "Predict News":
    st.header("Prediction")
    with st.form("Prediction from"):
        title = st.text_area("Enter the title of your text")
        text = st.text_area("Enter your text")
        submitted = st.form_submit_button("Submit")
    if submitted:
        df = pd.DataFrame({'title':[title], 'text': [text]})
        df = add_features(df)
        df["full_text"] = df["title"] + " " + df["text"]
        df = df.drop(["title, text"], axis=1)
        st.text(df.head())
        
    

