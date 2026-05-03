import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, brier_score_loss

def run_ablation_study():
    print("🔬 Initializing Multi-Variant Stress Model Ablation Study...")
    df = pd.read_csv('data/processed/stress_hardened_pool.csv', low_memory=False)
    
    # Define Variant Feature Subsets
    all_features = [c for c in df.columns if c not in ['Target', 'DatasetSource', 'priority', 'user_id', 'Employee_Id', 'id']]
    
    # 1. Variant B: No Engineered Features
    engineered = ['RecentStressSpike', 'RecoveryFailureScore', 'StressLoad', 'BurnoutRiskCalc', 'WeeklyStressTrend', 'ConsecutivePoorSleepDays', 'HighWorkloadFrequency']
    
    # 2. Variant C: No Proxies (Proxies are subset of Engineered)
    proxies = ['WeeklyStressTrend', 'ConsecutivePoorSleepDays', 'HighWorkloadFrequency']
    
    # 3. Variant E: Target-Blind (No WorkStress or EmotionalExhaustion)
    leaky_inputs = ['WorkStress', 'EmotionalExhaustion', 'WorkStress_missing', 'EmotionalExhaustion_missing']
    
    variants = {
        'Variant A (Full)': all_features,
        'Variant B (No Eng)': [f for f in all_features if f not in engineered],
        'Variant C (No Proxies)': [f for f in all_features if f not in proxies],
        'Variant D (Cross-Domain)': [f for f in all_features if not f.startswith('DatasetSource')],
        'Variant E (Target-Blind)': [f for f in all_features if f not in leaky_inputs],
        'Variant F (Minimal)': ['SleepHours', 'ScreenTime', 'WorkHours', 'Age', 'ExperienceYears']
    }
    
    results = []
    
    for v_name, v_features in variants.items():
        print(f"🏗️  Training {v_name} (Features: {len(v_features)})...")
        
        # Prepare
        X = df[v_features]
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
        
        # Train LGBM (representative of ensemble)
        model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train_s, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        results.append({
            'Variant': v_name,
            'Features': len(v_features),
            'AUC': roc_auc_score(y_test, y_prob),
            'Recall': recall_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Brier': brier_score_loss(y_test, y_prob)
        })
        
        # Save Variant E specifically if it's the target-blind candidate
        if 'Target-Blind' in v_name:
            joblib.dump(model, 'models/stress_variant_E_blind.pkl')
            joblib.dump(v_features, 'models/stress_variant_E_features.pkl')

    results_df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/ablation_study_results.csv', index=False)
    
    print("\n🏆 Ablation Study Results:")
    print(results_df.to_string(index=False))
    
    with open('reports/ablation_diagnostic.md', 'w') as f:
        f.write("# Stress Model Ablation Study: Anti-Leakage Diagnostic\n\n")
        f.write("This report analyzes how performance shifts as potentially leaky or simulated features are removed.\n\n")
        f.write("```text\n")
        f.write(results_df.to_string(index=False))
        f.write("\n```\n")
        f.write("\n\n> [!TIP]\n")
        f.write("> **Variant E (Target-Blind)** represents the truest measure of behavioral learning without direct survey indicators.")

if __name__ == "__main__":
    run_ablation_study()
