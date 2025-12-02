import streamlit as st
import pandas as pd
import numpy as np
import json, pickle
from catboost import CatBoostClassifier

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Loan Default Prediction")
st.markdown("Predict the likelihood of loan default based on customer information")

@st.cache_resource
def load_artifacts():
    # Load model
    model = CatBoostClassifier()
    model.load_model("models/catboost_clean.cbm")
    
    # Load scaler if exists
    scaler = None
    try:
        with open("models/scaler.pkl","rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None
    
    # Load features order
    with open("models/features.json","r") as f:
        features = json.load(f)
    
    return model, scaler, features

model, scaler, all_features = load_artifacts()

# Based on correlation with TARGET from your heatmap
IMPORTANT_FEATURES = [
    "EXT_SOURCE_2",       # Correlation: -0.16 (strongest)
    "EXT_SOURCE_3",       # Correlation: -0.16 (strongest)
    "EXT_SOURCE_1",       # Correlation: -0.10
    "AGE_YEARS",          # Correlation: -0.08
    "CREDIT_GOODS_RATIO", # Correlation: +0.06
    "AMT_CREDIT",         # Important financial feature
    "AMT_INCOME_TOTAL",   # Important financial feature
    "AMT_ANNUITY",        # Important financial feature
]

# Sidebar
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.162, 0.01)
st.sidebar.markdown(f"**Current:** `{threshold}`")
st.sidebar.markdown("---")

st.sidebar.subheader("📊 Feature Importance")
st.sidebar.markdown("""
**Top features (correlation with TARGET):**
- EXT_SOURCE_2: -0.16
- EXT_SOURCE_3: -0.16  
- EXT_SOURCE_1: -0.10
- AGE_YEARS: -0.08
- CREDIT_GOODS_RATIO: +0.06
""")

# File uploader
uploaded = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])

# Tabs
tab1, tab2 = st.tabs(["👤 Single Prediction", "📊 Batch Processing"])

with tab1:
    st.header("Single Customer Prediction")
    
    # Create 3 columns for inputs
    col1, col2, col3 = st.columns(3)
    
    sample_values = {}
    
    with col1:
        st.subheader("External Scores")
        sample_values["EXT_SOURCE_1"] = st.slider(
            "External Score 1", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="External credit score 1 (0-1)"
        )
        
        sample_values["EXT_SOURCE_2"] = st.slider(
            "External Score 2", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="External credit score 2 (0-1)"
        )
        
        sample_values["EXT_SOURCE_3"] = st.slider(
            "External Score 3", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="External credit score 3 (0-1)"
        )
    
    with col2:
        st.subheader("Customer Info")
        sample_values["AGE_YEARS"] = st.number_input(
            "Age (Years)",
            min_value=18, max_value=100, value=35, step=1
        )
        
        sample_values["AMT_INCOME_TOTAL"] = st.number_input(
            "Total Income",
            min_value=0, value=150000, step=10000,
            help="Annual total income"
        )
        
        sample_values["AMT_CREDIT"] = st.number_input(
            "Loan Amount",
            min_value=0, value=500000, step=10000,
            help="Total credit/loan amount"
        )
    
    with col3:
        st.subheader("Financial Ratios")
        sample_values["AMT_ANNUITY"] = st.number_input(
            "Loan Annuity",
            min_value=0, value=25000, step=1000,
            help="Annual loan payment"
        )
        
        sample_values["CREDIT_GOODS_RATIO"] = st.slider(
            "Credit to Goods Ratio",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help="Credit amount / Goods price ratio"
        )
        
        # Default value for other features
        st.markdown("---")
        default_value = st.number_input(
            "Default value for other features",
            value=0.0,
            step=0.1,
            help="Value for all non-important features"
        )
    
    # Predict button
    st.markdown("---")
    if st.button("🎯 Predict Default Risk", type="primary", use_container_width=True):
        # Create full sample
        full_sample = {}
        
        # Add important features from inputs
        for feature in IMPORTANT_FEATURES:
            if feature in sample_values:
                full_sample[feature] = sample_values[feature]
            else:
                full_sample[feature] = 0.0
        
        # Add all other features with default value
        for feature in all_features:
            if feature not in IMPORTANT_FEATURES:
                full_sample[feature] = default_value
        
        # Create dataframe
        df_sample = pd.DataFrame([full_sample])
        X = df_sample[all_features].astype(float)
        
        # Apply scaling if needed
        if scaler:
            X = scaler.transform(X)
        
        # Make prediction
        probability = model.predict_proba(X)[:,1][0]
        prediction = 1 if probability >= threshold else 0
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        # Create result columns
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            # Color code based on risk
            if probability < 0.3:
                color = "green"
                risk_level = "LOW RISK"
            elif probability < 0.7:
                color = "orange"
                risk_level = "MEDIUM RISK"
            else:
                color = "red"
                risk_level = "HIGH RISK"
            
            st.metric(
                "Default Probability", 
                f"{probability:.1%}",
                delta=risk_level,
                delta_color="off"
            )
        
        with result_col2:
            if prediction == 1:
                st.error("🚨 HIGH DEFAULT RISK")
                st.markdown(f"**Recommendation:** Reject application")
            else:
                st.success("✅ LOW DEFAULT RISK")
                st.markdown(f"**Recommendation:** Approve application")
        
        with result_col3:
            st.metric("Decision Threshold", f"{threshold:.1%}")
        
        # Visual gauge
        st.markdown("**Risk Level:**")
        
        # Create custom progress bar
        gauge_html = f"""
        <div style="background: #f0f2f6; border-radius: 10px; padding: 10px; margin: 10px 0;">
            <div style="background: linear-gradient(90deg, #00C851 0%, #ffbb33 50%, #ff4444 100%); 
                        height: 25px; border-radius: 5px; position: relative;">
                <div style="position: absolute; left: {probability*100}%; top: -5px; 
                            width: 3px; height: 35px; background: black; 
                            transform: translateX(-50%);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.9em;">
                <span>0% (Low Risk)</span>
                <span>50%</span>
                <span>100% (High Risk)</span>
            </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)
        
        # Decision explanation
        st.markdown("**Decision Logic:**")
        if prediction == 1:
            st.error(f"**High default risk predicted** because probability ({probability:.1%}) ≥ threshold ({threshold:.1%})")
        else:
            st.success(f"**Low default risk predicted** because probability ({probability:.1%}) < threshold ({threshold:.1%})")

with tab2:
    st.header("Batch Prediction")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ File loaded: {uploaded.name} ({len(df)} rows)")
        
        # Check for missing features
        missing = [f for f in all_features if f not in df.columns]
        if missing:
            st.error(f"❌ Missing {len(missing)} required features")
            with st.expander("Show missing features"):
                for f in missing:
                    st.write(f"`{f}`")
        else:
            X = df[all_features].astype(float)
            
            if scaler:
                X = scaler.transform(X)
            
            # Get predictions
            probabilities = model.predict_proba(X)[:,1]
            predictions = (probabilities >= threshold).astype(int)
            
            # Add to dataframe
            df["default_probability"] = probabilities
            df["high_risk_prediction"] = predictions
            
            # Show statistics
            st.subheader("📊 Batch Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Total Applications", len(df))
            
            with stat_col2:
                high_risk = predictions.sum()
                st.metric("High Risk Count", high_risk)
            
            with stat_col3:
                high_risk_pct = (high_risk / len(df)) * 100
                st.metric("High Risk %", f"{high_risk_pct:.1f}%")
            
            with stat_col4:
                avg_prob = probabilities.mean()
                st.metric("Avg Probability", f"{avg_prob:.1%}")
            
            # Show important columns in preview
            st.subheader("🔍 Preview")
            
            # Important columns to show
            preview_cols = ["high_risk_prediction", "default_probability"]
            preview_cols.extend([f for f in IMPORTANT_FEATURES if f in df.columns])
            
            # Limit preview rows
            st.dataframe(df[preview_cols].head(), use_container_width=True)
            
            # Download button
            st.markdown("---")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Results",
                data=csv,
                file_name="loan_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
    else:
        st.info("👆 Upload a CSV file in the sidebar")
        
        # Show required features
        with st.expander("📋 Required Features in CSV"):
            st.write(f"**Total features needed:** {len(all_features)}")
            st.write("**Most important features:**")
            
            for i, feature in enumerate(IMPORTANT_FEATURES):
                correlation = {
                    "EXT_SOURCE_2": "-0.16",
                    "EXT_SOURCE_3": "-0.16",
                    "EXT_SOURCE_1": "-0.10",
                    "AGE_YEARS": "-0.08",
                    "CREDIT_GOODS_RATIO": "+0.06",
                }.get(feature, "N/A")
                
                st.write(f"- **{feature}** (correlation: {correlation})")

# Footer
st.markdown("---")
st.caption(f"Model: CatBoost | Features: {len(all_features)} total | Key features: {len(IMPORTANT_FEATURES)} shown")