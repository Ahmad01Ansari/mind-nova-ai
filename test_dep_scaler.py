import joblib
scaler = joblib.load('models/depression_scaler.pkl')
try:
    print("Scaler features:", scaler.feature_names_in_)
except Exception as e:
    print("No feature names in", e)
    print("Scaler shape:", scaler.n_features_in_)
