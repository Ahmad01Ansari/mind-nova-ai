import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(name, model, X_test, y_test):
    """
    Step 8: Evaluation Metrics
    """
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For Neural Networks or models without predict_proba
        y_prob = model.predict(X_test).flatten()
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_summary_table(metrics_list):
    summary_df = pd.DataFrame(metrics_list)
    # Sort by priority: Recall > F1 > ROC-AUC
    summary_df = summary_df.sort_values(by=["Recall", "F1 Score", "ROC-AUC"], ascending=False)
    return summary_df
