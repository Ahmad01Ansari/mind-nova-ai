import joblib
model = joblib.load('models/depression_model_final.pkl')
try:
    print("Features expected (deprecated):", model.feature_names_in_)
except Exception as e:
    print("No feature_names_in_:", e)
try:
    print("Num features expected:", model.n_features_in_)
except Exception as e:
    print("No n_features_in_:", e)
