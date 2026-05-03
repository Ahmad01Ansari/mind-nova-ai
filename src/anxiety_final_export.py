import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath('src'))

import anxiety_preprocess as preprocess
import anxiety_feature_engineering as engineering
import anxiety_train as training
import anxiety_tune as tuning
import anxiety_evaluate as evaluate

def export_final_model():
    DATA_PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("🚀 Starting Final Export Pipeline...")
    
    # 1. Load and Preprocess
    df = preprocess.load_data(DATA_PATH)
    df_clean = preprocess.clean_data(df)
    
    # 2. Feature Engineering
    df_feat = engineering.engineer_features(df_clean)
    (X_A, y_A), (X_B, y_B) = engineering.get_feature_versions(df_feat)
    
    # Scale Features (using Version B for production)
    X_scaled, scaler = engineering.scale_features(X_B)
    
    # 3. Train-Test Split & SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_B, test_size=0.2, random_state=42, stratify=y_B)
    X_res, y_res = training.solve_imbalance(X_train, y_train)
    
    # 4. Hyperparameter Tuning (Top Performer: XGBoost)
    print("⌛ Tuning XGBoost for maximum Recall...")
    final_model, best_params = tuning.tune_xgboost(X_res, y_res)
    
    # 5. Evaluate
    metrics, y_pred, y_prob = evaluate.evaluate_model("Final Production Model", final_model, X_test, y_test)
    print("\n✅ Final Model Performance:")
    print(pd.DataFrame([metrics]))
    
    # 6. Save Artifacts (Step 12)
    print(f"\n💾 Saving artifacts to {MODEL_DIR}...")
    joblib.dump(final_model, os.path.join(MODEL_DIR, 'anxiety_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(X_B.columns.tolist(), os.path.join(MODEL_DIR, 'selected_features.pkl'))
    
    # Save a small report snippet
    with open(os.path.join(MODEL_DIR, 'model_metadata.txt'), 'w') as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['F1 Score']:.4f}\n")
    
    print("✨ Export Complete.")

if __name__ == "__main__":
    export_final_model()
