import pandas as pd
import numpy as np
import json
import os

def stream_synthetic_json(path, sample_size=150000):
    print(f"📦 Streaming Synthetic JSON: {os.path.basename(path)}...")
    df = pd.read_json(path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df

def unify_burnout_data(mapping_path='config/burnout_schema_mapping.json'):
    print("🧹 Initializing Burnout Unification Engine (Dual-Model Mode)...")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    all_dfs = []
    
    # Behavioral features to mask for Clinical Model backbone
    mask_features = ['SlackActivity', 'MeetingParticipation', 'EmailSentiment', 'OvertimeHours']
    
    for filename, meta in mapping.items():
        path = os.path.join('data/raw', filename)
        if not os.path.exists(path):
            continue
            
        source_name = meta['source_name']
        
        # 0. DROP STUDENT DATA AND WFH_CORPORATE FOR EXTERNAL LOSO AUDIT
        if source_name in ['Student_Burnout', 'DASS42_Psychiatric', 'WFH_Corporate']:
            print(f"🗑️ Dropping noise/holdout source: {source_name}")
            continue
            
        print(f"📥 Processing Source: {source_name}...")
        
        if filename.endswith('.json'):
            df = stream_synthetic_json(path, sample_size=150000)
        else:
            sep = '\t' if filename == 'data.csv' else ','
            df = pd.read_csv(path, sep=sep)
        
        # 1. Column Translation
        col_map = meta['column_map']
        df = df.rename(columns=col_map)
        
        # 2. Target Generation
        if meta['target_logic'] == 'direct_binary':
            df['RiskCategory'] = df['target_raw'].astype(int)
        elif meta['target_logic'] == 'burnout_level_map':
            df['RiskCategory'] = df['target_raw'].str.strip().str.capitalize().isin(['High', 'Severe']).astype(int)
        elif meta['target_logic'] == 'burnout_score_threshold':
            df['RiskCategory'] = (df['target_raw'] >= 60).astype(int)
        elif meta['target_logic'] == 'burnout_frequency_map':
            df['RiskCategory'] = df['target_raw'].str.strip().str.capitalize().isin(['Often', 'Always', 'Frequently']).astype(int)
        elif meta['target_logic'] == 'burnout_threshold_0.6':
            df['RiskCategory'] = (df['target_raw'] >= 0.6).astype(int)
            
        # 3. Standardize Features
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize().map({'Male': 0, 'Female': 1}).fillna(0.5)
            
        if 'StressLevel' in df.columns:
            stress_map = {'Low': 2.5, 'Medium': 5.0, 'Moderate': 5.0, 'High': 7.5, 'Severe': 10.0, 'Extreme': 10.0}
            df['StressLevel'] = df['StressLevel'].apply(lambda x: stress_map.get(str(x).strip().capitalize(), x)).astype(float)

        if 'SleepQuality' in df.columns:
            sleep_map = {'Poor': 2.5, 'Average': 5.0, 'Good': 7.5, 'Excellent': 10.0}
            df['SleepQuality'] = df['SleepQuality'].apply(lambda x: sleep_map.get(str(x).strip().capitalize(), x)).astype(float)

        # 5. Add Source Tracking
        df['DatasetSource'] = source_name

        # 6. Schema Alignment
        common_cols = [
            'WorkHours', 'OvertimeHours', 'SleepHours', 'ScreenTime', 'BreakFrequency', 
            'StressLevel', 'SleepQuality', 'JobSatisfaction', 'PhysicalActivity', 'ExperienceYears',
            'SlackActivity', 'MeetingParticipation', 'EmailSentiment', 'WorkloadScore', 'PerformanceScore',
            'RiskCategory', 'DatasetSource', 'Age', 'Gender'
        ]
        
        present_cols = [c for c in common_cols if c in df.columns]
        df = df[present_cols]
        all_dfs.append(df)

        # 4. Create Masked Duplicate for Clinical Backbone (Only for Synthetic)
        if source_name == 'Synthetic_Employee':
            print("🎭 Generating Clinical Backbone from Synthetic Data...")
            clinical_df = df.copy() # Slice is already done on df
            for feat in mask_features:
                if feat in clinical_df.columns:
                    clinical_df[feat] = np.nan
            clinical_df['DatasetSource'] = 'Synthetic_Clinical_Backbone'
            all_dfs.append(clinical_df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates()
    
    os.makedirs('data/merged', exist_ok=True)
    full_df.to_csv('data/merged/merged_burnout_data.csv', index=False)
    print(f"✅ Unified Burnout Data Ready (Dual-Model Enriched). Rows: {len(full_df)}")
    return full_df

if __name__ == "__main__":
    unify_burnout_data()
