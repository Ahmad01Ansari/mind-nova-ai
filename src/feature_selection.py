import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

def perform_feature_selection(processed_path='data/processed/processed_depression_data.csv'):
    df = pd.read_csv(processed_path)
    
    # Exclude non-features
    X = df.drop(columns=['RiskCategory', 'DatasetSource'])
    y = df['RiskCategory']
    
    # Handle NaNs for calculation (Feature Selection needs clean data)
    X_filled = X.fillna(X.median())
    
    print("📊 Computing Feature Importance Scores...")
    
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_filled, y, random_state=42)
    mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # 2. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_filled, y)
    rf_results = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Combined Audit
    print("\n--- Top Features (Mutual Information) ---")
    print(mi_results.head(10))
    
    print("\n--- Top Features (Random Forest) ---")
    print(rf_results.head(10))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    mi_results.plot(kind='barh')
    plt.title("Feature Importance (Mutual Information)")
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/feature_importance_mi.png')
    
    return mi_results, rf_results

if __name__ == "__main__":
    perform_feature_selection()
