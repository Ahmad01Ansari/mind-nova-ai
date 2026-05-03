import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.utils import class_weight

def train_deterioration_model_recovered():
    print("🏗️  Initializing Recovered Deterioration Sequence Training (Variant G)...")
    path = 'data/processed/deterioration_features_final.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    
    # 1. PREPARE TRAINING POOL
    # Feature Selection: Final Recovery Set (Anchor Triad + Escalation + Hardened Trends)
    features = [
        'MoodMini', 'SleepMini', 'WorkloadMini', 
        'EscalationVelocity_Mini', # COMPOSITE INTERACTION
        'SleepDecline_7D', 'ConsecutiveRiskDays',
        'RecoveryDeficitSlope', 'MoodVolatility', 'BurnoutAcceleration'
    ]
    
    X = df[features]
    y = df['Target']
    split = df['Split']
    
    # Split into Train/Val/Test based on the 'Split' column (Domain Priming preserved)
    X_train = X[split == 'train']
    y_train = y[split == 'train']
    
    X_test = X[split == 'test']
    y_test = y[split == 'test']
    
    # 2. DATA PIPELINE
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # 3. COMPUTE SPECIFIC CLASS WEIGHTS (Restoration Protocol)
    # Target: 0=Stable, 1=Mild, 2=Signif, 3=Crisis
    class_weights_dict = {
        0: 1.0,
        1: 2.0,
        2: 4.0,
        3: 10.0 # Extreme Crisis Priority
    }
    
    sample_weights = y_train.map(class_weights_dict)
    
    # 4. MODEL TRAINING
    print(f"🔥 Training Recovered XGBoost (Weighted) on {len(X_train)} samples...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        objective='multi:softprob',
        num_class=4,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # 5. EVALUATION
    y_prob = model.predict_proba(X_test_scaled)
    y_pred = model.predict(X_test_scaled)
    
    print("\n📊 Classification Report (90% Blind Holdout):")
    print(classification_report(y_test, y_pred, target_names=['Stable', 'Mild', 'Signif', 'Crisis']))
    
    # Calculate Macro AUC
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"📈 Recovered Macro ROC-AUC: {auc:.4f}")
    except:
        pass
        
    # Check Crisis Recall specifically
    crisis_rec = recall_score(y_test, y_pred, labels=[3], average=None)[0]
    print(f"🚩 Recovered Crisis Recall: {crisis_rec:.4f} (Target: > 0.80)")
    
    # 6. EXPORT
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/deterioration_model_recovered.pkl')
    joblib.dump(scaler, 'models/deterioration_scaler_recovered.pkl')
    joblib.dump(imputer, 'models/deterioration_imputer_recovered.pkl')
    joblib.dump(features, 'models/deterioration_features_recovered.pkl')
    
    print("\n✅ Recovery Training complete. Production artifacts exported to /models")

if __name__ == "__main__":
    train_deterioration_model_recovered()
