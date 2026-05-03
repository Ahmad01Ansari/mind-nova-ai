import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_nn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def perform_source_specific_imputation(df):
    """
    Imputes missing values using the median of each specific source.
    """
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude binary flags from imputation
    numeric_cols = [c for c in numeric_cols if not c.endswith('_missing')]
    
    for source in df['DatasetSource'].unique():
        idx = df['DatasetSource'] == source
        for col in numeric_cols:
            if df_imputed.loc[idx, col].isnull().any():
                source_median = df_imputed.loc[idx, col].median()
                if pd.isna(source_median):
                    source_median = df[col].median()
                df_imputed.loc[idx, col] = df_imputed.loc[idx, col].fillna(source_median)
    
    return df_imputed

def train_depression_models_weighted(data_path='data/processed/processed_depression_data.csv'):
    print("🚀 Initializing Final Recovery Training (Missingness Flags + 10x Weights)...")
    df = pd.read_csv(data_path)
    
    # 1. Source-Specific Imputation
    df = perform_source_specific_imputation(df)
    
    # 2. Prepare Data
    X_raw = df.drop(columns=['RiskCategory'])
    y = df['RiskCategory']
    
    # 3. One-Hot Encoding Source
    print("🏷️ Encoding DatasetSource...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    source_ohe = encoder.fit_transform(X_raw[['DatasetSource']])
    source_feature_names = encoder.get_feature_names_out(['DatasetSource'])
    
    source_df = pd.DataFrame(source_ohe, columns=source_feature_names)
    X = pd.concat([X_raw.drop(columns=['DatasetSource']).reset_index(drop=True), source_df], axis=1)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/depression_source_encoder.pkl')
    
    # Stratified Split
    strat_key = y.astype(str) + "_" + df['DatasetSource']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat_key)
    
    # Find weights (weights only on train)
    source_labels_train = encoder.inverse_transform(X_train[source_feature_names]).flatten()
    
    weights = []
    for s in source_labels_train:
        if s == 'University_Student': weights.append(10.0)
        elif s == 'Global_2025': weights.append(6.0)
        elif s == 'Student_Depression': weights.append(4.0)
        else: weights.append(1.0)
    weights = np.array(weights)
    
    # 4. Final Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/depression_scaler.pkl')
    
    # 5. Train Models
    print("🌲 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=8, random_state=42)
    xgb.fit(X_train_scaled, y_train, sample_weight=weights)
    
    print("🌿 Training LightGBM...")
    lgbm = LGBMClassifier(n_estimators=250, random_state=42)
    lgbm.fit(X_train_scaled, y_train, sample_weight=weights)
    
    print("🧠 Training Neural Network...")
    nn = build_nn(X_train_scaled.shape[1])
    nn.fit(X_train_scaled, y_train, sample_weight=weights, epochs=50, batch_size=128, verbose=0, validation_split=0.1)
    
    # 6. Evaluation
    models = {'XGBoost': xgb, 'LightGBM': lgbm, 'NeuralNet': nn}
    results = []
    
    for name, model in models.items():
        if name == 'NeuralNet':
            probs = model.predict(X_test_scaled).flatten()
            preds = (probs > 0.5).astype(int)
        else:
            probs = model.predict_proba(X_test_scaled)[:, 1]
            preds = model.predict(X_test_scaled)
            
        auc = roc_auc_score(y_test, probs)
        recall = recall_score(y_test, preds)
        print(f"[{name}] AUC: {auc:.4f} | Recall: {recall:.4f}")
        results.append({'Model': name, 'AUC': auc, 'Recall': recall})
        
        if name != 'NeuralNet': joblib.dump(model, f'models/depression_{name.lower()}.pkl')
        else: model.save(f'models/depression_{name.lower()}.keras') # Using recommended .keras format
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    train_depression_models_weighted()
