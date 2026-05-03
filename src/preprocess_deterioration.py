import pandas as pd
import numpy as np
import os
import random

def preprocess_deterioration():
    print("🚀 Initializing Multi-Source Timeline Preprocessing...")
    
    merged_timelines = []
    
    # --- RESOURCE 1: 14-Day Depression Symptoms ---
    print("📥 Processing Natural Depression Timelines...")
    dep_path = 'data/raw/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv'
    if os.path.exists(dep_path):
        dep = pd.read_csv(dep_path)
        # Assuming user_id is col 2, mood/phq are the features
        # Standardize: Map phq sum to Depression score 0-10
        phq_cols = [c for c in dep.columns if 'phq' in c.lower()]
        dep['Depression'] = dep[phq_cols].sum(axis=1) / (len(phq_cols)*3) * 10
        dep['user_id'] = dep['user_id'].astype(str)
        
        # We need a timepoint. Assuming rows are sequential per user
        dep['timepoint'] = dep.groupby('user_id').cumcount()
        
        dep_mapped = pd.DataFrame({
            'user_id': dep['user_id'] + "_dep",
            'timepoint': dep['timepoint'],
            'Depression': dep['Depression'],
            'Source': 'Depression_14D'
        })
        merged_timelines.append(dep_mapped)

    # --- RESOURCE 2: WFH Burnout (10 days per user) ---
    print("📥 Processing WFH Burnout Timelines...")
    wfh_path = 'data/raw/work_from_home_burnout_dataset.csv'
    if os.path.exists(wfh_path):
        wfh = pd.read_csv(wfh_path)
        # user_id, work_hours, screen_time, burnout_label
        # Map day to timepoint (Assuming 10 entries per user)
        wfh['timepoint'] = wfh.groupby('user_id').cumcount()
        
        wfh_mapped = pd.DataFrame({
            'user_id': wfh['user_id'].astype(str) + "_wfh",
            'timepoint': wfh['timepoint'],
            'Workload': wfh['work_hours'] / 1.5, # 15 hours max?
            'ScreenTime': wfh['screen_time_hours'] / 1.5,
            'Source': 'WFH_Burnout'
        })
        merged_timelines.append(wfh_mapped)

    # --- RESOURCE 3: Synthetic Drift (Cross-Sectional Growth) ---
    print("🧪 Generating Synthetic Behavioral Drifts...")
    # Using Healthcare Workforce as a seed
    seed_path = 'data/raw/Healthcare Workforce Mental Health Dataset.csv'
    if os.path.exists(seed_path):
        seed = pd.read_csv(seed_path).head(1000) # Use 1000 users as seeds
        drift_sequences = []
        for idx, row in seed.iterrows():
            uid = f"seed_{idx}_hc"
            base_stress = row['Stress Level']
            base_sleep = 7.0 
            
            # Generate 14 days of drift
            # 70% chance of worsening, 30% stable
            drift_type = "worsening" if random.random() < 0.7 else "stable"
            
            for t in range(14):
                if drift_type == "worsening":
                    # Slow exponential increase
                    current_stress = base_stress + (t * random.uniform(0.1, 0.4))
                    current_sleep = base_sleep - (t * random.uniform(0.1, 0.2))
                else:
                    # Random walk around baseline
                    current_stress = base_stress + random.uniform(-0.5, 0.5)
                    current_sleep = base_sleep + random.uniform(-0.3, 0.3)
                
                drift_sequences.append({
                    'user_id': uid,
                    'timepoint': t,
                    'Stress': np.clip(current_stress, 0, 10),
                    'Sleep': np.clip(current_sleep, 0, 10),
                    'Source': 'Synthetic_Drift'
                })
        merged_timelines.append(pd.DataFrame(drift_sequences))

    # --- FINAL MERGE ---
    print("🧬 Concatenating all timelines...")
    df_final = pd.concat(merged_timelines, ignore_index=True)
    
    # --- USER-LEVEL SPLIT (40/60 Domain Priming) ---
    depression_users = list(df_final[df_final['Source'] == 'Depression_14D']['user_id'].unique())
    other_users = list(df_final[df_final['Source'] != 'Depression_14D']['user_id'].unique())
    
    random.seed(42)
    random.shuffle(depression_users)
    random.shuffle(other_users)
    
    # 40% of Depression for Training (Priming), 60% for Holdout (Test)
    dep_split_idx = int(len(depression_users) * 0.4)
    dep_train_users = depression_users[:dep_split_idx]
    dep_test_users = depression_users[dep_split_idx:]
    
    # Split survivors (WFH + Synthetic) into train/val
    train_size = int(len(other_users) * 0.8)
    other_train_users = other_users[:train_size]
    other_val_users = other_users[train_size:]
    
    # Combine
    final_train = list(other_train_users) + list(dep_train_users)
    final_val = list(other_val_users) 
    final_test = list(dep_test_users)
    
    df_final['Split'] = 'train'
    df_final.loc[df_final['user_id'].isin(final_val), 'Split'] = 'val'
    df_final.loc[df_final['user_id'].isin(final_test), 'Split'] = 'test'
    
    print(f"✅ Preprocessing complete. Split Strategy: 40/60 Priming")
    print(f"   Train: {len(final_train)} (Includes {len(dep_train_users)} Depression users)")
    print(f"   Val:   {len(final_val)}")
    print(f"   Test:  {len(final_test)} (60% Depression Holdout)")
    
    os.makedirs('data/merged', exist_ok=True)
    df_final.to_csv('data/merged/deterioration_timelines_raw.csv', index=False)
    print("💾 Saved to data/merged/deterioration_timelines_raw.csv")

if __name__ == "__main__":
    preprocess_deterioration()
