import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import recall_score, precision_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_variant(path, name):
    print(f"\n🏗️  Training Variant: {name} (Source: {path})...")
    df = pd.read_csv(path)
    
    # Define features and target (Common intersection across all sources)
    features = ['WorkHours', 'SleepHours', 'ScreenTime', 'StressLevel']
    # Filter features that exist in the Pool
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df['RiskCategory']
    
    # Train Ensemble (Simplified for speed during experiment, matching production logic)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    lgbm = LGBMClassifier(n_estimators=100, num_leaves=31, random_state=42)
    
    clf = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm)],
        voting='soft'
    )
    
    clf.fit(X, y)
    
    save_path = f'models/experiments/burnout_variant_{name}.pkl'
    os.makedirs('models/experiments', exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"✅ Saved Variant {name} to {save_path}")
    return clf, features

def evaluate_model(clf, features, holdout_path, name, variant_name):
    if not os.path.exists(holdout_path):
        return None
        
    df = pd.read_csv(holdout_path)
    
    # Handle Scenario normalization inside evaluator if needed
    # (For Workplace_Holdout, we already mapped everything in recover_distributions.py)
    
    # Filter features
    X = df[features].fillna(5.0) # Neutral impute
    y_true = df['RiskCategory']
    
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        'Variant': variant_name,
        'Holdout': name,
        'AUC': roc_auc_score(y_true, y_prob),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Brier': brier_score_loss(y_true, y_prob)
    }
    return metrics

def run_recovery_experiment():
    variants = [
        {'name': 'BASELINE', 'path': 'data/experiment/Variant_A_Pool.csv'},
        {'name': 'NORMALIZED_MIXED', 'path': 'data/experiment/Variant_B_Pool.csv'},
        {'name': 'BALANCED_HYBRID', 'path': 'data/experiment/Variant_C_Pool.csv'}
    ]
    
    holdouts = [
        {'name': 'Workplace_Success_Anchor', 'path': 'data/experiment/Workplace_Holdout.csv'},
        {'name': 'Psychological_Stress_Test', 'path': 'data/raw/student_mental_health_burnout.csv'}
    ]
    
    all_results = []
    
    for v in variants:
        clf, feat_names = train_variant(v['path'], v['name'])
        
        for h in holdouts:
            print(f"📊 Evaluating {v['name']} on {h['name']}...")
            # For Student holdout, we need to map burnout_level and stress_level
            if h['name'] == 'Psychological_Stress_Test':
                df_h = pd.read_csv(h['path'])
                
                # Numeric Mapping for Categorical Stress
                if df_h['stress_level'].dtype == object:
                    stress_map = {'High': 8.5, 'Moderate': 5.0, 'Low': 2.5}
                    df_h['stress_level'] = df_h['stress_level'].str.strip().str.capitalize().map(stress_map).fillna(5.0)
                
                # Pre-map names
                std_mapping = {
                    'daily_study_hours': 'WorkHours',
                    'daily_sleep_hours': 'SleepHours',
                    'screen_time_hours': 'ScreenTime',
                    'stress_level': 'StressLevel'
                }
                df_h = df_h.rename(columns=std_mapping)
                df_h['RiskCategory'] = df_h['burnout_level'].str.strip().str.capitalize().isin(['High']).astype(int)
                
                # ENSURE NUMERIC for XGBoost
                for col in feat_names:
                    df_h[col] = pd.to_numeric(df_h[col], errors='coerce').fillna(5.0)
                
                # Save temp mapped student for evaluator
                temp_h = 'data/experiment/temp_student_mapped.csv'
                df_h.to_csv(temp_h, index=False)
                res = evaluate_model(clf, feat_names, temp_h, h['name'], v['name'])
            else:
                res = evaluate_model(clf, feat_names, h['path'], h['name'], v['name'])
                
            if res:
                all_results.append(res)
    
    # Save Report
    report_df = pd.DataFrame(all_results)
    report_df.to_csv('reports/recovery_experiment_metrics.csv', index=False)
    print("\n🏆 Recovery Experiment Complete. Summary Report:")
    print(report_df.to_string())

if __name__ == "__main__":
    run_recovery_experiment()
