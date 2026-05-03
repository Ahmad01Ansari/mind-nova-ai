import pandas as pd
import numpy as np
import joblib
import os
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from train import perform_source_specific_imputation

def calibrate_best_model(data_path='data/processed/processed_depression_data.csv'):
    print("⚖️ Initializing Model Calibration (OHE Aware)...")
    df = pd.read_csv(data_path)
    
    # 1. Source-Specific Imputation
    df = perform_source_specific_imputation(df)
    
    # 2. Prepare Data
    y = df['RiskCategory']
    X_raw = df.drop(columns=['RiskCategory'])
    
    # Load Encoder & Scaler
    encoder = joblib.load('models/depression_source_encoder.pkl')
    scaler = joblib.load('models/depression_scaler.pkl')
    
    # OHE Encoding
    source_ohe = encoder.transform(X_raw[['DatasetSource']])
    source_feature_names = encoder.get_feature_names_out(['DatasetSource'])
    source_df = pd.DataFrame(source_ohe, columns=source_feature_names)
    
    X = pd.concat([X_raw.drop(columns=['DatasetSource']).reset_index(drop=True), source_df], axis=1)
    
    # Split
    strat_key = y.astype(str) + "_" + df['DatasetSource']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat_key)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load Best Model (LightGBM)
    base_model = joblib.load('models/depression_lightgbm.pkl')
    
    # 3. Perform Calibration
    print("🔄 Training Calibrated Ensemble...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # 4. Save Final Artifacts
    joblib.dump(calibrated_model, 'models/depression_model_final.pkl')
    
    # 5. Evaluation of Calibration
    probs = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    fop, mpv = calibration_curve(y_test, probs, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(mpv, fop, marker='.', label='Calibrated LightGBM')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/calibration_curve.png')
    
    print("✅ Calibration complete. Final model saved to models/depression_model_final.pkl")
    return calibrated_model

if __name__ == "__main__":
    calibrate_best_model()
