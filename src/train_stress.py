import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score

def train_stress_model():
    print("🏗️  Initializing Weighted Stress Ensemble Training...")
    path = 'data/processed/stressed_engineered.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path, low_memory=False)
    
    # 1. SAMPLE WEIGHTING
    print("⚖️ Applying DatasetSource Weighting...")
    # Healthcare = 2.0x, Remote = 1.5x, Corporate = 1.5x, Student = 1.0x
    weight_map = {
        'Healthcare_Workforce': 2.0,
        'Remote_Work': 1.5,
        'Corporate_Stress': 1.5,
        'Student_Stress_Mon': 1.0,
        'DASS_Clinical': 1.0
    }
    df['sample_weight'] = df['DatasetSource'].map(weight_map).fillna(1.0)
    
    # 2. FEATURE SELECTION & PREPARATION
    # Drop non-inference columns
    X = df.drop(columns=['Target', 'DatasetSource', 'sample_weight'])
    y = df['Target']
    weights = df['sample_weight']
    
    # Identify Categorical vs Numerical
    X = pd.get_dummies(X, drop_first=True) # One-hot DatasetSource indicators if any leaked, or other strings
    
    # 3. SPLIT
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. PREPROCESSING PIPELINE
    print("🧹 Imputing and Scaling...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # 5. MODEL TRAINING (Weighted Ensemble)
    print("🔥 Training Weighted XGBoost, LightGBM, and Random Forest...")
    
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    lgbm = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    
    # Train individuals with weights
    xgb.fit(X_train_scaled, y_train, sample_weight=w_train)
    lgbm.fit(X_train_scaled, y_train, sample_weight=w_train)
    rf.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # 6. EVALUATION
    y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
    # Optimal threshold for Recall (as requested, explore 0.40 to 0.65)
    best_threshold = 0.50
    y_pred = (y_prob >= best_threshold).astype(int)
    
    print("\n📊 Initial Ensemble Metrics (Threshold=0.50):")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Recall:  {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")

    # 7. EXPORT
    print("💾 Exporting production artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble, 'models/stress_model.pkl')
    joblib.dump(scaler, 'models/stress_scaler.pkl')
    joblib.dump(imputer, 'models/stress_imputer.pkl')
    joblib.dump(list(X.columns), 'models/stress_features.pkl')
    
    # Save metadata
    with open('models/stress_metadata.txt', 'w') as f:
        f.write(f"Vesion: 1.0.0\n")
        f.write(f"FeatureCount: {len(X.columns)}\n")
        f.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
        
    print("✅ Training complete.")

if __name__ == "__main__":
    train_stress_model()
