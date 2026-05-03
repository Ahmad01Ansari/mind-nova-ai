import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import os

def check_source_performance():
    df = pd.read_csv('data/processed/deterioration_features_final.csv')
    model = joblib.load('models/deterioration_model.pkl')
    scaler = joblib.load('models/deterioration_scaler.pkl')
    imputer = joblib.load('models/deterioration_imputer.pkl')
    feats = joblib.load('models/deterioration_features.pkl')
    
    test_df = df[df['Split'] == 'test'].copy()
    X = scaler.transform(imputer.transform(test_df[feats]))
    y = test_df['Target']
    probs = model.predict_proba(X)
    
    global_auc = roc_auc_score(y, probs, multi_class='ovr')
    print(f"Global Test AUC: {global_auc:.4f}")
    
    for s in test_df['Source'].unique():
        mask = test_df['Source'] == s
        sub_y = y[mask]
        sub_probs = probs[mask]
        
        # Check if we have multiple classes to calculate AUC
        if len(sub_y.unique()) > 1:
            sub_auc = roc_auc_score(sub_y, sub_probs, multi_class='ovr')
            print(f"  Source {s:15} | AUC: {sub_auc:.4f} | Samples: {len(sub_y)}")
        else:
            print(f"  Source {s:15} | AUC: N/A (Single Class) | Samples: {len(sub_y)}")

if __name__ == "__main__":
    check_source_performance()
