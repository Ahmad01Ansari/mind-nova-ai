import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

def explain_stress_model():
    print("🧠 Initializing SHAP explainability Audit...")
    
    # Load Artifacts
    model_ensemble = joblib.load('models/stress_model.pkl')
    # For SHAP, we often use one of the underlying estimators if it's a VotingClassifier
    # or use the entire thing if it supports it.
    # Let's use the LGBM model for speed and clarity
    lgbm_model = model_ensemble.estimators_[1] # index 1 is LGBM
    
    scaler = joblib.load('models/stress_scaler.pkl')
    imputer = joblib.load('models/stress_imputer.pkl')
    features = joblib.load('models/stress_features.pkl')
    
    # Load Sample Data
    df = pd.read_csv('data/processed/stressed_engineered.csv', low_memory=False).head(2000)
    X = df.drop(columns=['Target', 'DatasetSource'])
    
    X = pd.get_dummies(X, drop_first=True)
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    
    X_processed = imputer.transform(X)
    X_scaled = scaler.transform(X_processed)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
    # 1. SHAP EXPLAINER
    print("🔍 Calculating SHAP Values (LGBM)...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    # Handle SHAP values format (binary classification)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    # 2. PLOT: Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_scaled_df, show=False)
    plt.title('MindNova Stress Risk: SHAP Feature Importance')
    os.makedirs('figures/stress_model', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/stress_model/shap_summary.png')
    plt.close()
    
    # 3. PLOT: Top 10 Features
    f_imp = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(shap_values_to_plot).mean(axis=0)
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(f_imp['Feature'].head(15), f_imp['Importance'].head(15), color='teal')
    plt.gca().invert_yaxis()
    plt.title('Top 15 Production Features: Behavioral Stress Model')
    plt.tight_layout()
    plt.savefig('figures/stress_model/feature_importance.png')
    plt.close()
    
    print("✅ Explainability Audit complete. Figures saved to figures/stress_model/")

if __name__ == "__main__":
    explain_stress_model()
