import pandas as pd
import numpy as np
import os
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
from burnout_ensemble import MindNovaBurnoutEnsemble

def perform_burnout_audit(data_path='data/processed/processed_burnout_data.csv'):
    print("🛡️ Initializing MindNova Burnout Source Audit (Dual-Ensemble)...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Load Ensemble
    ensemble = MindNovaBurnoutEnsemble()
    
    audit_results = []
    
    for source in df['DatasetSource'].unique():
        if pd.isna(source): continue
        if source == 'Synthetic_Clinical_Backbone': continue 
        
        print(f"📊 Auditing Source: {source}...")
        sub = df[df['DatasetSource'] == source].reset_index(drop=True)
        y_true = sub['RiskCategory'].values
        
        # Batch Inference
        results_df = ensemble.predict_risk_batch(sub)
        
        y_probs = results_df['burnout_probability'].values
        y_preds = results_df['needs_attention'].values
        model_used = results_df['selected_model'].values
            
        # Calculate Metrics
        auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
        recall = recall_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, zero_division=0)
        pos_rate = np.mean(y_true)
        
        primary_model = pd.Series(model_used).mode()[0]
        
        audit_results.append({
            'Source': source,
            'Records': len(sub),
            'Positives_%': f"{pos_rate:.1%}",
            'Model': primary_model,
            'Recall': f"{recall:.1%}",
            'Precision': f"{precision:.1%}",
            'ROC-AUC': round(auc, 3)
        })
        
    results_df = pd.DataFrame(audit_results)
    print("\n--- Burnout Source Audit Results (Ensemble Mode) ---")
    print(results_df.to_string(index=False))
    
    # Save Report
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/burnout_source_audit.csv', index=False)
    
if __name__ == "__main__":
    perform_burnout_audit()
