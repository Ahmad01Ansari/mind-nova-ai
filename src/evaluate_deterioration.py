import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_deterioration():
    print("📈 Initializing Deterioration Audit & Timeline Analysis...")
    
    # Load Artifacts
    model = joblib.load('models/deterioration_model.pkl')
    scaler = joblib.load('models/deterioration_scaler.pkl')
    imputer = joblib.load('models/deterioration_imputer.pkl')
    features = joblib.load('models/deterioration_features.pkl')
    
    df = pd.read_csv('data/processed/deterioration_features_final.csv')
    test_df = df[df['Split'] == 'test']
    
    X_test = test_df[features]
    X_test_p = imputer.transform(X_test)
    X_test_s = scaler.transform(X_test_p)
    
    # 1. SHAP TREND EXPLAINABILITY
    print("🧠 Calculating SHAP Trend Importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)
    
    # Crisis Class SHAP (Class 3)
    crisis_shap = shap_values[3] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(crisis_shap, X_test_s, feature_names=features, show=False)
    plt.title('Deterioration Drivers: Crisis Class SHAP')
    os.makedirs('figures/deterioration', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/deterioration/shap_summary_crisis.png')
    plt.close()

    # 2. SAMPLE USER TIMELINE AUDIT
    print("📽️ Generating User Timeline Visualization...")
    # Select a user from Depression source who hits Target 3
    sample_user = test_df[(test_df['Target'] == 3) & (test_df['Source'] == 'Depression_14D')]['user_id'].iloc[0]
    user_data = test_df[test_df['user_id'] == sample_user].sort_values('timepoint')
    
    X_u = user_data[features]
    y_u_prob = model.predict_proba(scaler.transform(imputer.transform(X_u)))
    
    # Extract prob of Crisis (Class 3)
    crisis_probs = y_u_prob[:, 3]
    
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    sns.lineplot(x=user_data['timepoint'], y=user_data['CompositeRisk'], ax=ax1, marker='o', label='Risk Magnitude', color='black')
    sns.lineplot(x=user_data['timepoint'], y=crisis_probs, ax=ax2, color='red', label='Crisis Probability', linestyle='--')
    
    # Fill background based on actual target
    for t in range(len(user_data)):
        target = user_data.iloc[t]['Target']
        color = 'white'
        if target == 1: color = 'yellow'
        if target == 2: color = 'orange'
        if target == 3: color = 'red'
        ax1.axvspan(user_data.iloc[t]['timepoint']-0.5, user_data.iloc[t]['timepoint']+0.5, color=color, alpha=0.2)

    ax1.set_title(f'User Timeline Audit: Deterioration Path ({sample_user})')
    ax1.set_xlabel('Day (Timepoint)')
    ax1.set_ylabel('Magnitude (0-10)')
    ax2.set_ylabel('Recall Probability')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/deterioration/user_timeline_audit.png')
    plt.close()
    
    print("✅ Evaluation complete. Artifacts saved to figures/deterioration/")

if __name__ == "__main__":
    evaluate_deterioration()
