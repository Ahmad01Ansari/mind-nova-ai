from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2']
    }
    search = RandomizedSearchCV(LogisticRegression(max_iter=2000), param_distributions=param_grid, n_iter=10, scoring='recall', cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=10, scoring='recall', cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_xgboost(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), param_distributions=param_grid, n_iter=10, scoring='recall', cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_lightgbm(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 62, 127],
        'max_depth': [-1, 10, 20, 30],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9]
    }
    search = RandomizedSearchCV(LGBMClassifier(random_state=42, verbose=-1), param_distributions=param_grid, n_iter=10, scoring='recall', cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
