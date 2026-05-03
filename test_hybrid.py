import joblib
feats = joblib.load('models/hybrid_features.pkl')
print("Hybrid features:", feats)
scaler = joblib.load('models/hybrid_scaler.pkl')
print("Hybrid scaler features:", scaler.feature_names_in_)
