import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath('src'))

import hybrid_preprocess as preprocess
import hybrid_feature_engineering as engineering
import hybrid_evaluate as evaluate

def run_representative_validation():
    DATA_PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    MODEL_PATH = 'models/binary_optimized_model.pkl'
    SCALER_PATH = 'models/hybrid_scaler.pkl'
    FEATURES_PATH = 'models/hybrid_features.pkl'
    
    print("🔬 Starting Representative Validation Suite...")
    
    # 1. Load & Process
    df = preprocess.load_data(DATA_PATH)
    df_labeled = preprocess.create_hybrid_labels(df)
    df_feat = engineering.engineer_hybrid_features(df_labeled)
    df_clean = preprocess.drop_diagnostic_features(df_feat)
    
    selected_features = joblib.load(FEATURES_PATH)
    X = df_clean[selected_features]
    y = df_clean['RiskCategory']
    
    # 2. Match the exact split from training (Random State 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Apply Scaler
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features)
    
    # 4. Load Model
    model = joblib.load(MODEL_PATH)
    
    # 5. Threshold Sweep [0.90 - 0.999]
    # Extreme search to find the 15-20% FPR intersection given high model confidence
    thresholds = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    print(f"\n⚡ Evaluating Thresholds: {thresholds}")
    results = evaluate.evaluate_binary_thresholds(model, X_test_scaled, y_test, thresholds=thresholds)
    
    print("\n📈 Representative Threshold Comparison Table:")
    print(results)
    
    # 6. Check Baseline
    print("\n⚠️ Model Discrimination Check (ROC-AUC):")
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    return results

if __name__ == "__main__":
    run_representative_validation()
