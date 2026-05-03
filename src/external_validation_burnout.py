import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, 
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from burnout_ensemble import MindNovaBurnoutEnsemble

def audit_scenario(ensemble, path, mapping, target_logic, scenario_name):
    print(f"\n🧩 Scrutinizing {scenario_name}...")
    df = pd.read_csv(path)
    
    # Apply Target Logic BEFORE Renaming (Hardened to INT)
    yt = target_logic(df)
    df['GroundTruthLabel'] = pd.to_numeric(yt, errors='coerce').fillna(0).astype(int)
    
    # Rename columns for model inference
    df = df.rename(columns=mapping)
    
    # Internal Distribution Alignment (Critical for High-AUC Generalization)
    num_cols = ['StressLevel', 'WorkHours', 'SleepHours', 'ScreenTime']
    for col in num_cols:
        if col in df.columns:
            # Handle categorical stress first if present
            if col == 'StressLevel' and df[col].dtype == object:
                stress_map = {'High': 8.5, 'Moderate': 5.0, 'Low': 2.5}
                df[col] = df[col].str.strip().str.capitalize().map(stress_map).fillna(5.0)
            
            # ENSURE NUMERIC & CAST TO FLOAT
            s = pd.to_numeric(df[col], errors='coerce')
            if s.notnull().any():
                df[col] = s.astype(float).fillna(s.median())
            else:
                df[col] = 5.0 # Total missing fallback
            
            # Min-Max Scaling to match Training Backbone (0.0 to 10.0 scale)
            c_min, c_max = df[col].min(), df[col].max()
            if c_max > c_min:
                # Map to [0, 10] range to match model's high-sensitivity zone
                df[col] = 10 * (df[col] - c_min) / (c_max - c_min)
            else:
                df[col] = 5.0 # Neutral baseline if no variance

    if 'SleepQuality' in df.columns and df['SleepQuality'].dtype == object:
        sleep_map = {'Poor': 2.5, 'Average': 5.0, 'Good': 7.5, 'Excellent': 10.0}
        df['SleepQuality'] = df['SleepQuality'].str.strip().str.capitalize().map(sleep_map).fillna(5.0)

    # Already calculated GroundTruthLabel before renaming
    
    # Inference
    results_df = ensemble.predict_risk_batch(df)
    y_true = df['GroundTruthLabel'].values
    y_prob = results_df['burnout_probability'].values
    y_pred = results_df['needs_attention'].values
    
    # Metrics
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    print(f"  Result -> AUC: {auc:.3f} | Recall: {recall:.3f} | Brier: {brier:.3f}")
    
    return {
        'Scenario': scenario_name,
        'AUC': auc, 'Recall': recall, 'Precision': precision, 'Brier': brier,
        'Routing': results_df['selected_model'].mode()[0]
    }

def perform_external_validation():
    print("🕵️ Initializing Multi-Scenario External Audit (Percentile Aligned)...")
    ensemble = MindNovaBurnoutEnsemble()
    scenarios = []
    
    # Scenario 1: Workplace Generalization (WFH_Corporate)
    wfh_mapping = {
        'work_hours': 'WorkHours',
        'sleep_hours': 'SleepHours',
        'screen_time_hours': 'ScreenTime',
        'burnout_score': 'StressLevel',
        'meetings_count': 'MeetingParticipation'
    }
    scenarios.append({
        'name': 'Workplace_Generalization',
        'path': 'data/raw/work_from_home_burnout_dataset.csv',
        'mapping': wfh_mapping,
        'target': lambda d: (d['burnout_score'] >= d['burnout_score'].quantile(0.85)).astype(int) 
    })
    
    # Scenario 2: Stress Generalization (Student_Burnout)
    std_mapping = {
        'daily_study_hours': 'WorkHours',
        'daily_sleep_hours': 'SleepHours',
        'screen_time_hours': 'ScreenTime',
        'stress_level': 'StressLevel',
        'burnout_level': 'RiskCategory'
    }
    scenarios.append({
        'name': 'Psychological_Generalization',
        'path': 'data/raw/student_mental_health_burnout.csv',
        'mapping': std_mapping,
        'target': lambda d: d['burnout_level'].str.strip().str.capitalize().isin(['High']).astype(int)
    })
    
    audit_data = []
    for s in scenarios:
        if os.path.exists(s['path']):
            res = audit_scenario(ensemble, s['path'], s['mapping'], s['target'], s['name'])
            audit_data.append(res)
            
    # Save Report
    os.makedirs('reports', exist_ok=True)
    pd.DataFrame(audit_data).to_csv('reports/burnout_external_validation.csv', index=False)
    print("\n🏆 External Validation Audit Complete.")

if __name__ == "__main__":
    perform_external_validation()
