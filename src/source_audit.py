import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, 
    brier_score_loss, confusion_matrix, roc_curve, 
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

def perform_source_audit(data_path='data/processed/processed_depression_data.csv'):
    print("🛡️ Initializing MindNova Source-Recovery Audit...")
    df = pd.read_csv(data_path)
    
    # Needs same imputation as training
    from train import perform_source_specific_imputation
    df = perform_source_specific_imputation(df)
    
    # 1. Setup Data & Pipeline
    y = df['RiskCategory']
    sources_all = df['DatasetSource']
    
    # Split
    strat_key = y.astype(str) + "_" + sources_all
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df.drop(columns=['RiskCategory']), y, test_size=0.2, random_state=42, stratify=strat_key)
    
    # Load Models & Encoders
    encoder = joblib.load('models/depression_source_encoder.pkl')
    scaler = joblib.load('models/depression_scaler.pkl')
    model = joblib.load('models/depression_model_final.pkl')
    
    # Process Test Set (OHE + Scale)
    source_test_ohe = encoder.transform(X_test_raw[['DatasetSource']])
    source_feature_names = encoder.get_feature_names_out(['DatasetSource'])
    source_test_df = pd.DataFrame(source_test_ohe, columns=source_feature_names)
    
    X_test_clean = pd.concat([X_test_raw.drop(columns=['DatasetSource']).reset_index(drop=True), source_test_df], axis=1)
    X_test_scaled = scaler.transform(X_test_clean)
    
    probs_all = model.predict_proba(X_test_scaled)[:, 1]
    sources_test = X_test_raw['DatasetSource'].values
    unique_sources = np.unique(sources_test)
    
    audit_results = []
    os.makedirs('reports/audit_plots', exist_ok=True)
    
    for source in unique_sources:
        print(f"📊 Auditing Source: {source}...")
        idx = (sources_test == source)
        y_sub = y_test.values[idx]
        probs_sub = probs_all[idx]
        
        if len(y_sub) < 10: continue
            
        preds_sub = (probs_sub >= 0.65).astype(int)
        recall = recall_score(y_sub, preds_sub, zero_division=0)
        precision = precision_score(y_sub, preds_sub, zero_division=0)
        auc = roc_auc_score(y_sub, probs_sub) if len(np.unique(y_sub)) > 1 else np.nan
        brier = brier_score_loss(y_sub, probs_sub)
        
        tn, fp, fn, tp = confusion_matrix(y_sub, preds_sub, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        audit_results.append({
            'Source': source,
            'Records': len(y_sub),
            'Positives_%': f"{y_sub.mean():.1%}",
            'Recall': f"{recall:.1%}",
            'Precision': f"{precision:.1%}",
            'ROC-AUC': f"{auc:.3f}",
            'FPR': f"{fpr:.1%}",
            'Brier': f"{brier:.4f}"
        })
        
    results_df = pd.DataFrame(audit_results)
    print("\n--- Final Recovery Audit Results ---")
    print(results_df.to_string(index=False))
    results_df.to_csv('reports/source_recovery_audit.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    perform_source_audit()
