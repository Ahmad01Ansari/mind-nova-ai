import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def source_specific_imputation(df):
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if not c.endswith('_missing') and c != 'RiskCategory']
    
    for source in df['DatasetSource'].unique():
        idx = df['DatasetSource'] == source
        for col in numeric_cols:
            if col in df_imputed.columns and df_imputed.loc[idx, col].isnull().any():
                source_median = df_imputed.loc[idx, col].median()
                if pd.isna(source_median):
                    source_median = df[col].median()
                df_imputed.loc[idx, col] = df_imputed.loc[idx, col].fillna(source_median)
    return df_imputed

def train_submodel(df, features, model_name, weights_dict=None):
    print(f"🛠️ Training Sub-model: {model_name}...")
    
    # 1. Impute & Prepare
    df = source_specific_imputation(df)
    X = df[features]
    y = df['RiskCategory']
    
    # Identify Source for Weighting
    sources = df['DatasetSource'].values
    weights = np.ones(len(y))
    if weights_dict:
        for source, w in weights_dict.items():
            weights[sources == source] = w
            
    # 2. Split
    # Since some sources might be very small, we use simple stratify by y
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, f'models/{model_name}_scaler.pkl')
    joblib.dump(features, f'models/{model_name}_features.pkl')
    
    # 4. Train Ensemble (LGBM & XGB)
    print(f"  🌲 [{model_name}] Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    print(f"  🌿 [{model_name}] Training LightGBM...")
    lgbm = LGBMClassifier(n_estimators=150, random_state=42, verbose=-1)
    lgbm.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # 5. Evaluate
    results = []
    for name, model in [('XGB', xgb), ('LGBM', lgbm)]:
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs > 0.5).astype(int)
        # For clinical model, we check 0.4 threshold for higher recall
        if 'clinical' in model_name.lower():
            preds = (probs > 0.4).astype(int)
            
        auc = roc_auc_score(y_test, probs)
        recall = recall_score(y_test, preds)
        print(f"  ✅ [{model_name}-{name}] AUC: {auc:.4f} | Recall: {recall:.4f}")
        results.append(model)
    
    # Save the best (LGBM usually better for large data)
    joblib.dump(lgbm, f'models/{model_name}_core.pkl')
    return {
        'model': lgbm,
        'scaler': scaler,
        'features': features
    }

def execute_dual_training(data_path='data/processed/processed_burnout_data.csv'):
    print("🚀 Initializing Dual-Core Burnout Training Pipeline...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Feature Sets (Synchronized with current post-pivot schema)
    behavioral_features = [
        'SlackActivity', 'MeetingParticipation', 'EmailSentiment', 'WorkloadScore', 
        'PerformanceScore', 'DigitalOverloadIndex', 'WorkloadIntensity',
        'OvertimeHours'
    ]
    common_features = [
        'WorkHours', 'SleepHours', 'ScreenTime', 'BreakFrequency', 
        'StressLevel', 'ExperienceYears', 'JobSatisfaction',
        'StressLoad', 'SleepDebt', 'RecoveryScore'
    ]
    missing_flags = [c for c in df.columns if c.endswith('_missing')]
    
    # 🟢 1. INTEGRATED MODEL (Model A)
    # Target: Users with integrated workplace data
    integrated_sources = ['Synthetic_Employee', 'Remote_Work']
    df_integrated = df[df['DatasetSource'].isin(integrated_sources)]
    feat_a = common_features + behavioral_features + missing_flags
    weights_a = {'Remote_Work': 10.0}
    train_submodel(df_integrated, feat_a, 'burnout_integrated', weights_a)
    
    # 🔴 2. CLINICAL MODEL (Model B)
    # Target: Healthcare, Corporate, and users without integrations
    clinical_sources = ['Synthetic_Clinical_Backbone', 'Healthcare_Workforce', 'WFH_Corporate', 'Remote_Work']
    df_clinical = df[df['DatasetSource'].isin(clinical_sources)]
    feat_b = common_features + [f for f in missing_flags if 'Slack' not in f and 'Meeting' not in f]
    weights_b = {'Healthcare_Workforce': 15.0, 'WFH_Corporate': 5.0, 'Remote_Work': 5.0}
    train_submodel(df_clinical, feat_b, 'burnout_clinical', weights_b)
    
    print("\n🏆 Dual-Core Training Complete. All artifacts saved to /models.")

if __name__ == "__main__":
    execute_dual_training()
