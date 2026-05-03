import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train import perform_source_specific_imputation

def evaluate_depression_model(data_path='data/processed/processed_depression_data.csv'):
    print("📈 Initializing Final Production Evaluation (OHE + Flags Aware)...")
    df = pd.read_csv(data_path)
    
    # 1. Pipeline Matching
    df = perform_source_specific_imputation(df)
    y = df['RiskCategory']
    X_raw = df.drop(columns=['RiskCategory'])
    
    # Stratified Split (Match training)
    strat_key = y.astype(str) + "_" + df['DatasetSource']
    _, X_test_raw, _, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=strat_key)
    
    # Load Models & Encoders
    encoder = joblib.load('models/depression_source_encoder.pkl')
    scaler = joblib.load('models/depression_scaler.pkl')
    model = joblib.load('models/depression_model_final.pkl')
    
    # Transform Test Set
    source_ohe = encoder.transform(X_test_raw[['DatasetSource']])
    source_feature_names = encoder.get_feature_names_out(['DatasetSource'])
    source_df = pd.DataFrame(source_ohe, columns=source_feature_names)
    
    X_test_clean = pd.concat([X_test_raw.drop(columns=['DatasetSource']).reset_index(drop=True), source_df], axis=1)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # 2. Probabilities
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    brier = brier_score_loss(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    
    # 3. Threshold Comparison
    thresholds = [0.25, 0.35, 0.50, 0.65, 0.80]
    comparison = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        comparison.append({
            'Threshold': t,
            'Recall': f"{recall:.1%}",
            'Precision': f"{precision:.1%}",
            'FPR': f"{fpr:.1%}"
        })
    
    comp_df = pd.DataFrame(comparison)
    print("\n--- Global Threshold Optimization Table ---")
    print(comp_df.to_string(index=False))
    
    # Recommend
    best_prod = comp_df[comp_df['Threshold'] == 0.65].iloc[0]
    print(f"\n💡 Production Recommendation: Threshold {best_prod['Threshold']}")
    
    return comp_df

if __name__ == "__main__":
    evaluate_depression_model()
