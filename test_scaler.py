import joblib
scaler = joblib.load('models/burnout_scaler.pkl')
features = scaler.feature_names_in_
print("len:", len(features))
has_academic = "AcademicStress" in features
has_break = "BreakFrequency" in features
print("has AcademicStress:", has_academic)
print("has BreakFrequency:", has_break)
