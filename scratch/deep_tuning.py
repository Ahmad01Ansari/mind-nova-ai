import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add src to path
sys.path.append(os.path.abspath('src'))

import anxiety_preprocess as preprocess
import anxiety_feature_engineering as engineering
import anxiety_train as training
import anxiety_evaluate as evaluate

def deep_tune():
    DATA_PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    df = preprocess.load_data(DATA_PATH)
    df_clean = preprocess.clean_data(df)
    df_feat = engineering.engineer_features(df_clean)
    
    versions = {
        "Version A (All)": engineering.get_feature_versions(df_feat)[0],
        "Version B (Behavioral)": engineering.get_feature_versions(df_feat)[1]
    }
    
    results = []
    
    for name, vdata in versions.items():
        print(f"\n🚀 Intensive tuning for {name}...")
        X, y = vdata
        X_scaled, _ = engineering.scale_features(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        X_res, y_res = training.solve_imbalance(X_train, y_train)
        
        # Intensive param grid
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': [100, 300, 500, 1000],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
            'min_child_weight': [1, 3, 5]
        }
        
        # Increase n_iter to 50 for depth
        search = RandomizedSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            param_distributions=param_grid,
            n_iter=50,
            scoring='recall',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_res, y_res)
        best_model = search.best_estimator_
        
        # Evaluate on test set
        metrics, _, _ = evaluate.evaluate_model(f"Deep Tuned XGB - {name}", best_model, X_test, y_test)
        results.append(metrics)
        print(f"Metrics for {name}: Accuracy: {metrics['Accuracy']:.4f}, Recall: {metrics['Recall']:.4f}")

    print("\n--- Final Performance Comparison ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    deep_tune()
