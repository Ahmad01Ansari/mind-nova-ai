import joblib
model = joblib.load('models/burnout_xgboost.pkl')
try:
    print("Features:", model.get_booster().feature_names)
except Exception as e:
    print(e)
