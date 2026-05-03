import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def tune_hybrid_xgboost(X_train, y_train):
    """
    Tune XGBoost for multiclass risk classification.
    """
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    xgb = XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, eval_metric='mlogloss')
    
    search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=10, 
                                scoring='recall_weighted', cv=3, random_state=42, n_jobs=-1)
    
    print("Tuning XGBoost...")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_hybrid_lightgbm(X_train, y_train):
    """
    Tune LightGBM for multiclass risk classification.
    """
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500],
        'num_leaves': [31, 64, 128],
        'feature_fraction': [0.7, 0.8, 0.9]
    }
    
    lgbm = LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1)
    
    search = RandomizedSearchCV(lgbm, param_distributions=param_grid, n_iter=10, 
                                scoring='recall_weighted', cv=3, random_state=42, n_jobs=-1)
    
    print("Tuning LightGBM...")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
