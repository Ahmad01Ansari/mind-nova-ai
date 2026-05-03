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

def train_deterioration_model():
    print("🏗️  Initializing Deterioration Sequence Training (Variant A: XGBoost)...")
    path = 'data/processed/deterioration_features_final.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    
    # 1. PREPARE TRAINING POOL
    # Drop rows without enough window history (timepoint < 7) if we want pure maturity
    # But for MVP, we let imputer handle early cold-start
    
    # Feature Selection: Exclude non-inference columns
    features = [
        'SleepDecline_7D', 'ConsecutiveRiskDays',
        'RecoveryDeficitSlope', 'MoodVolatility', 'BurnoutAcceleration'
    ]
    
    X = df[features]
    y = df['Target']
    split = df['Split']
    
    # Split into Train/Val/Test based on the 'Split' column (User-level split preserved)
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
    
    # 3. COMPUTE CLASS WEIGHTS (Safety First: prioritizing Crisis/3)
    # We want to give Crisis (3) higher weight
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(zip(np.unique(y_train), weights))
    # Boost Crisis (3) even further for safety
    class_weights_dict[3] *= 1.5 
    
    sample_weights = y_train.map(class_weights_dict)
    
    # 4. MODEL TRAINING
    print(f"🔥 Training XGBoost on {len(X_train)} samples...")
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
    
    print("\n📊 Classification Report (Blind Test):")
    print(classification_report(y_test, y_pred, target_names=['Stable', 'Mild', 'Signif', 'Crisis']))
    
    # Calculate Macro AUC
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"📈 Macro ROC-AUC: {auc:.4f}")
    except:
        pass
        
    # Check Crisis Recall specifically
    crisis_rec = recall_score(y_test, y_pred, labels=[3], average=None)[0]
    print(f"🚩 Crisis Recall: {crisis_rec:.4f} (Target: > 0.90)")
    
    # 6. EXPORT
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/deterioration_model.pkl')
    joblib.dump(scaler, 'models/deterioration_scaler.pkl')
    joblib.dump(imputer, 'models/deterioration_imputer.pkl')
    joblib.dump(features, 'models/deterioration_features.pkl')
    
    print("\n✅ Training complete. Production artifacts exported to /models")

if __name__ == "__main__":
    train_deterioration_model()
