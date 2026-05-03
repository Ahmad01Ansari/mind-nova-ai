import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

import anxiety_preprocess as preprocess
import anxiety_feature_engineering as engineering
import anxiety_train as training
import anxiety_evaluate as evaluate

def run_test_pipeline():
    DATA_PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    
    print("Loading data...")
    df = preprocess.load_data(DATA_PATH)
    
    print("Cleaning data...")
    df_clean = preprocess.clean_data(df)
    
    print("Engineering features...")
    df_feat = engineering.engineer_features(df_clean)
    (X_A, y_A), (X_B, y_B) = engineering.get_feature_versions(df_feat)
    
    print("Splitting and Balancing...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
    X_res, y_res = training.solve_imbalance(X_train, y_train)
    
    print("Training Baselines...")
    models = training.train_baseline_models(X_res, y_res)
    
    print("Evaluating Top 2...")
    all_metrics = []
    for name in ["Random Forest", "XGBoost"]:
        metrics, _, _ = evaluate.evaluate_model(name, models[name], X_test, y_test)
        all_metrics.append(metrics)
        
    summary = evaluate.get_summary_table(all_metrics)
    print("\nModel Verification Summary:")
    print(summary)

if __name__ == "__main__":
    run_test_pipeline()
