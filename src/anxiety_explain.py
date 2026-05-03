import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_shap_plots(model, X_train, X_test, feature_names, save_prefix):
    """
    Step 10: SHAP summary and waterfall
    """
    # Sample if too large
    if len(X_test) > 100:
        X_sample = X_test.sample(100, random_state=42)
    else:
        X_sample = X_test
        
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.savefig(f"{save_prefix}_shap_summary.png", bbox_inches='tight')
    plt.close()
    
    # Waterfall plot for first sample
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(f"{save_prefix}_shap_waterfall.png", bbox_inches='tight')
    plt.close()

def generate_lime_explanation(model, X_train, X_test, feature_names, sample_idx=0):
    """
    Step 10: LIME explanation for a specific prediction
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        mode='classification'
    )
    
    # Handle both scikit-learn and Keras models
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    else:
        # For Keras models
        def keras_predict(X):
            probs = model.predict(X)
            return np.hstack([1-probs, probs])
        predict_fn = keras_predict
    
    exp = explainer.explain_instance(
        data_row=X_test.iloc[sample_idx].values,
        predict_fn=predict_fn
    )
    
    return exp
