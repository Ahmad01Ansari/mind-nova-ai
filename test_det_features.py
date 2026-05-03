import joblib
feats = joblib.load('models/deterioration_features_recovered.pkl')
print("Deterioration features:", feats)
