import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import shap

def select_by_mutual_info(X, y, k=15):
    """
    Select top K features based on Mutual Information.
    """
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_series.head(k).index.tolist()

def select_by_rfe(X, y, n_features=12):
    """
    Select top N features based on Recursive Feature Elimination with Random Forest.
    """
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
              n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()

def select_by_shap(model, X, k=15):
    """
    Select top K features based on SHAP importance.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Absolute SHAP values
    if isinstance(shap_values, list): # For multi-output models or multiclass
        shap_sum = np.abs(shap_values[1]).mean(axis=0)
    else: # For binary models
        shap_sum = np.abs(shap_values).mean(axis=0)
        
    shap_series = pd.Series(shap_sum, index=X.columns).sort_values(ascending=False)
    return shap_series.head(k).index.tolist()

def get_consensus_features(mi_feats, rfe_feats, shap_feats):
    """
    Identify features that appear in multiple selection strategies.
    """
    all_selected = mi_feats + rfe_feats + shap_feats
    counts = pd.Series(all_selected).value_counts()
    consensus = counts[counts >= 2].index.tolist()
    return consensus

if __name__ == "__main__":
    print("Feature Selection module ready.")
