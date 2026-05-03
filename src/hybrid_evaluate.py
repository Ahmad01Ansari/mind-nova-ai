import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.metrics import precision_recall_curve, roc_curve, auc

def evaluate_binary_thresholds(model, X_test, y_test, thresholds=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50]):
    """
    Evaluate multiple probability thresholds to maximize Recall.
    Added False Positive Rate (FPR) for operational load analysis.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        # Handle cases where confusion matrix might not be 2x2 (rare in this dataset)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0
            
        metrics = {
            "Threshold": t,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "FPR": fpr
        }
        results.append(metrics)
        
    return pd.DataFrame(results)

def plot_roc_curve(model, X_test, y_test, name="Model"):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {name}')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(model, X_test, y_test, name="Model"):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend(loc="lower left")
    plt.show()

def plot_binary_cm(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Needs Attention'],
                yticklabels=['Safe', 'Needs Attention'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def evaluate_binary_model(name, model, X_test, y_test, threshold=0.5):
    """
    Binary evaluation with custom threshold.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    return metrics, y_pred, y_prob

def plot_multiclass_cm(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Moderate', 'High'],
                yticklabels=['Low', 'Moderate', 'High'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def plot_calibration_curve_and_brier(model, X_test, y_test, name="Model"):
    """
    Evaluate how well predicted probabilities match empirical risk.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    brier = brier_score_loss(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='s', label=f'{name} (Brier: {brier:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    return brier

def plot_probability_distribution(model, X_test, y_test, name="Model"):
    """
    Visualize probability density split by actual class.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_df = pd.DataFrame({'Probability': y_prob, 'Actual': y_test})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=plot_df, x='Probability', hue='Actual', element='step', 
                 palette={0: 'blue', 1: 'red'}, common_norm=False, bins=25)
    plt.title(f'Probability Distribution by Class - {name}')
    plt.xlabel('Predicted Probability of Risk')
    plt.ylabel('Density')
    plt.legend(title='Actual Risk', labels=['Needs Attention (1)', 'Safe (0)'])
    plt.show()

def get_hybrid_summary_table(metrics_list):
    return pd.DataFrame(metrics_list)
