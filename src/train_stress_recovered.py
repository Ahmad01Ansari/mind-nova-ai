import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, brier_score_loss

def train_recovered_model():
    print("🏗️  Initializing Stress Recovery Training (Variant G: Hybrid)...")
    df = pd.read_csv('data/processed/stressed_engineered.csv', low_memory=False)
    
    # Define Feature Sets
    # Variant G = Behavioral + Mini Features
    features_G = [c for c in df.columns if c not in ['Target', 'DatasetSource', 'priority', 'user_id', 'Employee_Id', 'id', 'Employee_Id.1']]
    
    print(f"📊 Training on {len(features_G)} features: {features_G}")
    
    X = df[features_G]
    X = pd.get_dummies(X, drop_first=True)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_p = imp.fit_transform(X_train)
    X_test_p = imp.transform(X_test)
    X_train_s = scaler.fit_transform(X_train_p)
    X_test_s = scaler.transform(X_test_p)
    
    # Train LGBM (representative)
    print("🔥 Fitting Hybrid Model...")
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train_s, y_train)
    
    # 1. EVALUATION
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    
    print(f"\n📊 Variant G (Recovered Hybrid) Metrics:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Recall: {rec:.4f}")
    print(f"   Precision: {pre:.4f}")
    print(f"   Brier: {brier:.4f}")
    
    # 2. SHAP IMPORTANCE ( distributed check)
    print("\n🧠 Analyzing Feature Importance (Distributed Check)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)
    
    # Handle SHAP values list/ndarray
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    f_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(shap_values_to_plot).mean(axis=0)
    }).sort_values(by='Importance', ascending=False)
    
    print("Top 10 Features (Importance):")
    print(f_imp.head(10).to_string(index=False))
    
    # 3. EXPORT
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stress_model_recovered.pkl')
    joblib.dump(scaler, 'models/stress_scaler_recovered.pkl')
    joblib.dump(imp, 'models/stress_imputer_recovered.pkl')
    joblib.dump(list(X.columns), 'models/stress_features_recovered.pkl')
    
    # Plot SHAP for walkthrough
    os.makedirs('figures/stress_recovery', exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_test_p, feature_names=X.columns, show=False)
    plt.title('Stress Recovery (Variant G): SHAP Summary')
    plt.tight_layout()
    plt.savefig('figures/stress_recovery/shap_recovered.png')
    plt.close()

if __name__ == "__main__":
    train_recovered_model()
