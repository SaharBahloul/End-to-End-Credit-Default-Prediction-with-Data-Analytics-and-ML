
import streamlit as st
import pandas as pd
import numpy as np
import json, pickle
from catboost import CatBoostClassifier

st.set_page_config(page_title="CatBoost Inference", layout="centered")
st.title("CatBoost Inference")

@st.cache_resource
def load_artifacts():
    # load model
    model = CatBoostClassifier()
    model.load_model("models/catboost_clean.cbm")
    # load scaler if exists
    scaler = None
    try:
        with open("models/scaler.pkl","rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None
    # load features order
    with open("models/features.json","r") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_artifacts()

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.162, 0.01)  # default = your best_thr

uploaded = st.file_uploader("Upload CSV with features", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df.head())
    # align columns
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Missing features: {missing}")
    else:
        X = df[features].astype(float)
        if scaler: X = scaler.transform(X)
        proba = model.predict_proba(X)[:,1]
        df["proba_positive"] = proba
        df["pred_positive"] = (proba >= threshold).astype(int)
        st.dataframe(df.head())
        st.download_button("Download results CSV", df.to_csv(index=False).encode("utf-8"), file_name="preds.csv")
else:
    st.info("Upload a CSV to get batch predictions (columns must include training features).")
    st.write("Or predict a single sample (first 20 features).")
    sample = {}
    for f in features[:20]:
        sample[f] = st.number_input(f, value=0.0, format="%.6f", key=f)
    if st.button("Predict single sample"):
        df_s = pd.DataFrame([sample])
        # fill other features with 0
        for f in features[20:]:
            df_s[f] = 0.0
        Xs = df_s[features].astype(float)
        if scaler: Xs = scaler.transform(Xs)
        p = model.predict_proba(Xs)[:,1][0]
        st.write("Probability:", float(p))
        st.write("Prediction (threshold {:.3f}):".format(threshold), int(p >= threshold))
