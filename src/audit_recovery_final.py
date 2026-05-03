import pandas as pd
import joblib
import numpy as np
import shap

def audit_final():
    model = joblib.load('models/deterioration_model_recovered.pkl')
    scaler = joblib.load('models/deterioration_scaler_recovered.pkl')
    imputer = joblib.load('models/deterioration_imputer_recovered.pkl')
    features = joblib.load('models/deterioration_features_recovered.pkl')
    
    df = pd.read_csv('data/processed/deterioration_features_final.csv')
    test_df = df[df['Split'] == 'test'].head(100)
    X = scaler.transform(imputer.transform(test_df[features]))
    
    explainer = shap.TreeExplainer(model)
    shap_v = explainer.shap_values(X)
    
    # SHAP for Class 3 (Crisis)
    if isinstance(shap_v, list):
        # Multiclass SHAP list of [class0, class1, class2, class3]
        target_shap = shap_v[3]
    else:
        # Array shape [samples, features, classes]
        target_shap = shap_v[:, :, 3]
        
    importances = np.abs(target_shap).mean(axis=0)
    total = importances.sum()
    
    f_imp = pd.DataFrame({
        'feat': features,
        'imp': importances,
        'pct': (importances/total)*100
    }).sort_values('imp', ascending=False)
    
    print("\n🚩 RECOVERY AUDIT: Feature Importance (Crisis Class)")
    print(f_imp.to_string(index=False))

if __name__ == "__main__":
    audit_final()
