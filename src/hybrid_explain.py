import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def generate_hybrid_shap_plots(model, X_train, X_test, feature_names):
    """
    Generate SHAP summary and waterfall plots for multiclass.
    """
    print("Generating SHAP plots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Multiclass)")
    plt.show()
    
    return shap_values

def generate_hybrid_lime_explanation(model, X_train, X_test, feature_names, sample_idx):
    """
    Generate LIME explanation for a specific user.
    """
    print(f"Generating LIME for user index {sample_idx}...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['Low', 'Moderate', 'High'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        data_row=X_test.iloc[sample_idx],
        predict_fn=model.predict_proba
    )
    
    return exp

def get_risk_drivers(model, feature_names):
    """
    Identify overall risk-driving factors from feature importance.
    """
    # For tree models, feature_importances_ is a good proxy for global impact
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    # In mental health, top importance items are often high-stress indicators
    drivers = feat_imp.head(5).index.tolist()
    protective = feat_imp.tail(5).index.tolist()
    
    return drivers, protective

if __name__ == "__main__":
    # Test block
    print("Explainability module ready for multiclass analysis.")
