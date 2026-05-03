import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def solve_multiclass_imbalance(X_train, y_train):
    """
    Apply SMOTE to balance the 3 risk categories.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def train_hybrid_models(X_train, y_train):
    """
    Train Binary Classification models for MindNova.
    Using simple 3-layer NN as requested.
    """
    # Calculate scale_pos_weight for imbalance
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=spw, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1, is_unbalance=True)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

from sklearn.calibration import CalibratedClassifierCV

def train_calibrated_suite(X_train, y_train):
    """
    Train a suite of calibrated classifiers using 5-fold cross-validation.
    This ensures reliable probabilities and avoids overfitting the calibrator.
    """
    # Calculate scale_pos_weight for imbalance
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count
    
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=spw, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1, is_unbalance=True)
    }
    
    calibrated_models = {}
    for name, model in base_models.items():
        print(f"Training & Calibrating {name}...")
        # Calibrate using Isotonic Regression with 5-fold CV
        cal_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
        cal_model.fit(X_train, y_train)
        calibrated_models[name] = cal_model
        
    return calibrated_models

def train_3layer_neural_network(X_train, y_train, input_dim):
    """
    Step 7.5: 3-layer Neural Network for Binary Task
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
    return model

def save_hybrid_artifacts(model, scaler, feature_names, model_name='hybrid_risk_model'):
    """
    Step 13: Save final production artifacts.
    """
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'hybrid_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'hybrid_features.pkl'))
    print(f"Artifacts saved to {MODEL_DIR}")

if __name__ == "__main__":
    from hybrid_preprocess import load_data, create_hybrid_labels, drop_diagnostic_features
    from hybrid_feature_engineering import engineer_hybrid_features, scale_features
    
    PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    df = create_hybrid_labels(load_data(PATH))
    df_feat = engineer_hybrid_features(df)
    df_final = drop_diagnostic_features(df_feat)
    
    X = df_final.drop(columns=['RiskCategory'])
    y = df_final['RiskCategory']
    
    X_scaled, scaler = scale_features(X, X.columns)
    X_res, y_res = solve_multiclass_imbalance(X_scaled, y)
    
    models = train_hybrid_models(X_res, y_res)
    # Save the first one for now
    save_hybrid_artifacts(models["XGBoost"], scaler, X.columns.tolist())
