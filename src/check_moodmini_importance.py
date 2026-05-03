import pandas as pd
import joblib
import numpy as np
import shap

def audit_importance():
    model = joblib.load('models/deterioration_model_recovered.pkl')
    scaler = joblib.load('models/deterioration_scaler_recovered.pkl')
    imputer = joblib.load('models/deterioration_imputer_recovered.pkl')
    features = joblib.load('models/deterioration_features_recovered.pkl')
    
    df = pd.read_csv('data/processed/deterioration_features_final.csv')
    test_df = df[df['Split'] == 'test'].head(2000)
    X = scaler.transform(imputer.transform(test_df[features]))
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Class 3 (Crisis) Importance
    if isinstance(shap_values, list):
        shap_v = shap_values[3]
    else:
        shap_v = shap_values
        
    importances = np.abs(shap_v).mean(axis=0)
    total = importances.sum()
    
    f_imp = pd.DataFrame({'feat': features, 'imp': importances, 'pct': (importances/total)*100})
    print("\n🚩 SHAP Importance Audit (Crisis Class):")
    print(f_imp.sort_values(by='imp', ascending=False).to_string(index=False))

if __name__ == "__main__":
    audit_importance()
