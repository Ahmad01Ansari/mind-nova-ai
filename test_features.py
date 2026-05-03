import joblib
model = joblib.load('models/burnout_xgboost.pkl')
try:
    print("Features expected:", model.get_booster().feature_names)
except:
    try:
        print("Features expected:", model.feature_names_in_)
    except Exception as e:
        print(e)
