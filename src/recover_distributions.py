import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def recover_distributions():
    print("🚀 Initializing Burnout Distribution Recovery...")
    os.makedirs('data/experiment', exist_ok=True)
    
    # 1. LOAD DATA
    merged_path = 'data/merged/merged_burnout_data.csv'
    wfh_path = 'data/raw/work_from_home_burnout_dataset.csv'
    
    df = pd.read_csv(merged_path)
    wfh_raw = pd.read_csv(wfh_path)
    
    # 2. CARVE OUT WFH HOLDOUT (20%)
    print(f"📥 Carving out 20% Holdout from WFH_Corporate ({len(wfh_raw)} records)...")
    # For safety, ensure we don't include the holdout in any training variants
    wfh_train, wfh_holdout = train_test_split(wfh_raw, test_size=0.20, random_state=42)
    
    # Save Holdout
    # We need to map WFH raw features for consistent evaluation later
    # Mapping based on previous external_validation_burnout.py logic
    wfh_mapping = {
        'work_hours': 'WorkHours',
        'sleep_hours': 'SleepHours',
        'screen_time_hours': 'ScreenTime',
        'burnout_score': 'StressLevel'
    }
    wfh_holdout_mapped = wfh_holdout.rename(columns=wfh_mapping)
    # Important: Scaling Holdout Stress to match Training 0-10 range
    wfh_holdout_mapped['StressLevel'] = (wfh_holdout_mapped['StressLevel'] / 10.0).clip(0, 10.0)
    
    # Balanced target logic for evaluation: Medium/High burnout_risk -> 1
    wfh_holdout_mapped['RiskCategory'] = wfh_holdout['burnout_risk'].str.strip().str.capitalize().isin(['High', 'Medium']).astype(int)
    wfh_holdout_mapped.to_csv('data/experiment/Workplace_Holdout.csv', index=False)
    
    # 3. DISTRIBUTION NORMALIZATION (Synthetic -> Real-World Scale)
    print("⚖️ Normalizing Synthetic Distributions...")
    
    # We want Synthetic Stress mean 0.79 -> ~5.0
    # Linear scale: New = Alpha + Beta * Old
    # If 0.0 -> 2.0 and 1.0 -> 8.0, Mean 0.79 -> 2 + 0.79 * 6 = 6.74 (Maybe slightly high, let's target mean 5.0)
    # 0.0 -> 1.0, 1.0 -> 6.0 => Mean 0.79 -> 1 + 0.79 * 5 = 4.95 (Perfect!)
    
    def normalize_source_features(source_df, source_name):
        if 'Synthetic' in source_name:
            # StressLevel: Match Real-World Range [1, 6] for synthetic base
            source_df['StressLevel'] = 1.0 + (source_df['StressLevel'] * 5.0)
            
            # WorkHours: Synthetic is mostly 8.4mean (8-9 range), 
            # Real-world Remote is also 8+ mean. No deep change needed, but ensure parity.
            pass
            
            # ScreenTime: Synthetic ScreenTime was often low or missing.
            # Real-world is often 8-12 hours.
            if 'ScreenTime' in source_df.columns:
                 # If screen time exists but is low, shift it up
                 if source_df['ScreenTime'].max() <= 1.0:
                     source_df['ScreenTime'] = source_df['ScreenTime'] * 12.0
        return source_df

    sources = df['DatasetSource'].unique()
    normalized_dfs = []
    for s in sources:
        s_df = df[df['DatasetSource'] == s].copy()
        s_df = normalize_source_features(s_df, s)
        normalized_dfs.append(s_df)
    
    df_normalized = pd.concat(normalized_dfs)
    
    # 4. GENERATE VARIANTS
    
    # Variant A: Original Pool (Already exists in data/merged/merged_burnout_data.csv)
    # But we should ensure it's saved in experiment dir for parity
    df.to_csv('data/experiment/Variant_A_Pool.csv', index=False)
    
    # Variant B: Normalized Mixed Pool (300k+ records)
    df_normalized.to_csv('data/experiment/Variant_B_Pool.csv', index=False)
    
    # Variant C: Balanced Hybrid (100% Real + 16k Normalized Synthetic)
    print("🎨 Sculpting Variant C (Balanced Hybrid)...")
    real_data = df_normalized[~df_normalized['DatasetSource'].str.contains('Synthetic')]
    synthetic_data = df_normalized[df_normalized['DatasetSource'].str.contains('Synthetic')]
    
    synthetic_sample = synthetic_data.sample(n=16000, random_state=42)
    variant_c_pool = pd.concat([real_data, synthetic_sample])
    variant_c_pool.to_csv('data/experiment/Variant_C_Pool.csv', index=False)
    
    print(f"✅ Distribution Recovery Complete.")
    print(f"   Holdout: {len(wfh_holdout_mapped)} records")
    print(f"   Variant B: {len(df_normalized)} records")
    print(f"   Variant C: {len(variant_c_pool)} records")

if __name__ == "__main__":
    recover_distributions()
