import joblib
scaler = joblib.load('models/burnout_scaler.pkl')
feats = joblib.load('models/burnout_clinical_features.pkl')

s_feats = list(scaler.feature_names_in_)
if s_feats == feats:
    print("Match!")
else:
    print("Mismatch!")
    print("In Scaler but not in Feats:", set(s_feats) - set(feats))
    print("In Feats but not in Scaler:", set(feats) - set(s_feats))
