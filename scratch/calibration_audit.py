import joblib
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath('src'))

import hybrid_preprocess as preprocess
import hybrid_feature_engineering as engineering
import hybrid_evaluate as evaluate

def generate_calibration_audit():
    DATA_PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    MODEL_PATH = 'models/binary_optimized_model.pkl'
    SCALER_PATH = 'models/hybrid_scaler.pkl'
    FEATURES_PATH = 'models/hybrid_features.pkl'
    
    print("📊 Generating Probability Calibration Audit...")
    
    # 1. Load data & Preprocess
    df = preprocess.load_data(DATA_PATH)
    df_labeled = preprocess.create_hybrid_labels(df)
    df_feat = engineering.engineer_hybrid_features(df_labeled)
    df_clean = preprocess.drop_diagnostic_features(df_feat)
    
    selected_features = joblib.load(FEATURES_PATH)
    X = df_clean[selected_features]
    y = df_clean['RiskCategory']
    
    # Use SAME test split state 42 for consistency
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Load model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features)
    
    # 3. Generate Plots
    # Note: These functions call plt.show()
    # In this environment, they will be captured as artifacts if I save them.
    
    print(f"Brier Score: {evaluate.brier_score_loss(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")
    
    # Probability Distribution Plot
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    plot_df = pd.DataFrame({'Probability': y_prob, 'Actual': y_test})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=plot_df, x='Probability', hue='Actual', element='step', 
                 palette={0: 'blue', 1: 'red'}, common_norm=False, bins=25)
    plt.title('Calibrated Probability Distribution by Class')
    plt.xlabel('Predicted Probability of Risk')
    plt.ylabel('Density')
    plt.savefig('models/calibration_distribution.png')
    print("Saved: models/calibration_distribution.png")
    
    # Calibration Curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='s', label='Calibrated XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.savefig('models/calibration_curve.png')
    print("Saved: models/calibration_curve.png")

if __name__ == "__main__":
    generate_calibration_audit()
