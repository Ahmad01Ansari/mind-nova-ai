import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def explain_depression_model(data_path='data/processed/processed_depression_data.csv'):
    print("🔮 Initializing SHAP Explainability Suite...")
    df = pd.read_csv(data_path)
    
    # 1. Prepare Data
    X = df.drop(columns=['RiskCategory', 'DatasetSource'])
    
    # Load Preprocessing
    imputer = joblib.load('models/depression_imputer.pkl')
    scaler = joblib.load('models/depression_scaler.pkl')
    X_processed = scaler.transform(imputer.transform(X))
    
    # Load Final Model (Calibrated)
    model_wrapper = joblib.load('models/depression_model_final.pkl')
    # For SHAP, we need the base estimator. 
    # CalibratedClassifierCV stores estimators in .calibrated_classifiers_
    base_model = model_wrapper.calibrated_classifiers_[0].estimator
    
    # 2. SHAP Calculation
    # Use a background sample for TreeExplainer
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_processed[:1000]) # Sample of 1000 for speed
    
    # 3. Visualizations
    os.makedirs('figures', exist_ok=True)
    
    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_processed[:1000], feature_names=X.columns, show=False)
    plt.title("SHAP Feature Importance (Overall Influence)")
    plt.tight_layout()
    plt.savefig('figures/shap_summary.png')
    plt.close()
    
    # Waterfall Plot for a high-risk user
    # Find a user with RiskCategory=1
    high_risk_idx = df[df['RiskCategory'] == 1].index[0]
    plt.figure(figsize=(12, 8))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[high_risk_idx], 
                                           feature_names=X.columns, show=False)
    plt.title(f"Clinical Logic: High-Risk User Audit (ID: {high_risk_idx})")
    plt.tight_layout()
    plt.savefig('figures/shap_high_risk_audit.png')
    plt.close()
    
    print("✨ Explainability audit complete. Visuals saved to figures/")

if __name__ == "__main__":
    explain_depression_model()
