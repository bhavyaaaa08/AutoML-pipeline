import streamlit as st
import pandas as pd
import numpy as np
from nec_functions import run_automl_pipeline
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the page
st.set_page_config(page_title="AutoML Pipeline", layout="wide")

# Title and description
st.title("AutoML Pipeline")
st.write("""
This application automatically builds and evaluates machine learning models for your dataset.
Upload a CSV file, select the target column, and let the system do the rest!
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Show basic info
            st.subheader("Dataset Info")
            st.write(f"Shape: {df.shape}")
            st.write("First 5 rows:")
            st.dataframe(df.head())
            
            # Select target column
            target_col = st.selectbox("Select target column", df.columns)
            
            # Advanced options
            st.subheader("Advanced Options")
            n_trials = st.slider("Number of optimization trials", 10, 100, 30)
            
            if st.button("Run AutoML Pipeline"):
                with st.spinner("Running AutoML pipeline..."):
                    try:
                        results = run_automl_pipeline(df, target_col, n_trials=n_trials)
                        st.session_state.results = results
                        st.success("Pipeline completed successfully!")
                    except Exception as e:
                        st.error(f"Error running pipeline: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Main content area
if 'results' in st.session_state:
    results = st.session_state.results
    
    st.header("Model Results")
    
    # Show metrics based on task type
    if results['task_type'] == 'classification':
        st.subheader("Classification Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{results['metrics']['accuracy']:.4f}")
            st.text("Classification Report:")
            st.text(results['metrics']['classification_report'])
            
        with col2:
            st.image(BytesIO(base64.b64decode(results['metrics']['confusion_matrix'])), 
                     caption="Confusion Matrix")
            
            if results['metrics']['roc_curve']:
                st.image(BytesIO(base64.b64decode(results['metrics']['roc_curve'])), 
                         caption="ROC Curve")
    else:  # regression
        st.subheader("Regression Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R² Score", f"{results['metrics']['r2']:.4f}")
            st.metric("RMSE", f"{results['metrics']['rmse']:.4f}")
            st.metric("MAE", f"{results['metrics']['mae']:.4f}")
            
        with col2:
            st.image(BytesIO(base64.b64decode(results['metrics']['residual_plot'])), 
                     caption="Residual Plot")
            st.image(BytesIO(base64.b64decode(results['metrics']['actual_vs_predicted'])), 
                     caption="Actual vs Predicted")
    
    # Feature importance and SHAP plots
    st.header("Model Interpretability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results['feature_importance']:
            st.image(BytesIO(base64.b64decode(results['feature_importance'])), 
                     caption="Feature Importance")
    
    with col2:
        if results['shap_summary']:
            st.image(BytesIO(base64.b64decode(results['shap_summary'])), 
                     caption="SHAP Summary Plot")
    
    # Sample predictions with explanations
    st.header("Sample Predictions with Explanations")
    
    if results['X_test_processed'].shape[0] > 0:
        sample_idx = st.slider("Select sample to explain", 0, results['X_test_processed'].shape[0]-1, 0)
        
        if results['task_type'] == 'classification':
            pred = results['best_model'].predict(results['X_test_processed'][sample_idx:sample_idx+1])
            proba = results['best_model'].predict_proba(results['X_test_processed'][sample_idx:sample_idx+1])
            
            if results['label_encoder']:
                pred_class = results['label_encoder'].inverse_transform(pred)[0]
                class_probs = dict(zip(results['label_encoder'].classes_, proba[0]))
            else:
                pred_class = pred[0]
                class_probs = dict(zip(range(len(proba[0])), proba[0]))
            
            st.subheader(f"Prediction: {pred_class}")
            st.write("Class probabilities:")
            st.write(class_probs)
        else:  # regression
            pred = results['best_model'].predict(results['X_test_processed'][sample_idx:sample_idx+1])
            actual = results['y_test'][sample_idx]
            
            st.subheader(f"Prediction: {pred[0]:.4f}")
            st.write(f"Actual value: {actual:.4f}")
            st.write(f"Error: {abs(pred[0] - actual):.4f}")
        
        # Generate SHAP force plot for this sample
        from nec_functions import generate_shap_force_plot
        force_plot = generate_shap_force_plot(
            results['best_model'],
            results['X_test_processed'],
            results['X_test_processed'],
            results['feature_names'],
            sample_idx,
            results['task_type']
        )
        
        if force_plot:
            st.image(BytesIO(base64.b64decode(force_plot)), 
                     caption="SHAP Explanation for this prediction")