import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

def load_config(config_path='config/schema_mapping.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_dass(df, config):
    """
    Specific transformation for DASS-42 dataset.
    Sums the 14 Depression questions.
    """
    dep_cols = config['mappings']['data.csv']['dass_depression_cols']
    # DASS scores are shifted (1-4). Subtract 1 to get (0-3).
    df_dep = df[dep_cols].apply(lambda x: x - 1)
    df['target_raw'] = df_dep.sum(axis=1)
    # Threshold > 13 for Moderate+ Depression (Binary 1)
    df['RiskCategory'] = (df['target_raw'] > 13).astype(int)
    # Extract PHQ2 proxy from first two DASS-Dep items (Q3, Q5)
    df['PHQ2'] = (df[['Q3A', 'Q5A']].apply(lambda x: x - 1)).sum(axis=1).clip(0, 6)
    return df

def standardize_categorical_features(df):
    """
    Maps string-based lifestyle features to numeric scales.
    """
    # Sanitize Columns
    df.columns = [c.strip() for c in df.columns]
    
    # Sleep mapping (Case-insensitive)
    # Ensure keys are lowercase and stripped
    sleep_map = {
        '5-6 hours': 5.5,
        'less than 5 hours': 4.0,
        '7-8 hours': 7.5,
        'more than 8 hours': 9.5,
        '8-9 hours': 8.5,
        'others': 6.0
    }
    
    if 'SleepHours' in df.columns:
        # Save original as numeric where possible
        s = df['SleepHours'].astype(str).str.lower().str.strip()
        df['SleepHours'] = s.map(sleep_map)
        # If it was already numeric (but type object), the map might return NaN.
        # Fallback to pd.to_numeric for strings that might be digits
        df['SleepHours'] = df['SleepHours'].fillna(pd.to_numeric(s, errors='coerce'))
    
    # Financial Stress mapping
    stress_map = {'high': 8, 'moderate': 5, 'low': 2, 'no': 0}
    if 'FinancialStress' in df.columns:
        s = df['FinancialStress'].astype(str).str.lower().str.strip()
        df['FinancialStress'] = s.map(stress_map)
        df['FinancialStress'] = df['FinancialStress'].fillna(pd.to_numeric(s, errors='coerce'))
        
    return df

def merge_datasets(config_path='config/schema_mapping.json'):
    # Standardize to absolute paths to avoid confusion
    # BASE_DIR is the mind_nova_ai folder
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'schema_mapping.json')
    
    config = load_config(CONFIG_PATH)
    common_cols = config['common_features'] + ['RiskCategory', 'DatasetSource']
    final_dfs = []

    for filename, meta in config['mappings'].items():
        path = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️ Warning: {path} not found. Skipping...")
            continue
        
        # Determine separator
        sep = '\t' if filename == 'data.csv' else ','
        df = pd.read_csv(path, sep=sep)
        
        # Apply Mapping
        col_map = meta['column_map']
        df = df.rename(columns=col_map)
        
        # Standardize categorical lifestyle features
        df = standardize_categorical_features(df)
        
        # Apply specific logic
        if meta['target_logic'] == 'PHQ9_target':
            if meta['source_name'] == "PHQ9_14Day":
                # SUM all 9 items for target
                phq_cols = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']
                df['target_raw'] = df[phq_cols].sum(axis=1)
                # PHQ2 proxy for this source
                df['PHQ2'] = df[['phq1', 'phq2']].sum(axis=1).clip(0, 6)
            
            if 'target_raw' in df.columns:
                df['RiskCategory'] = (df['target_raw'] >= 10).astype(int)
        elif meta['target_logic'] == 'DASS_sum':
            df = preprocess_dass(df, config)
        elif meta['target_logic'] == 'direct_binary':
            # Target is already labeled, but ensure it's 0/1
            df['RiskCategory'] = df['RiskCategory'].astype(str).str.strip().map({
                '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0
            }).fillna(0)
        
        # Scaling
        if 'scaling' in meta:
            for col, factor in meta['scaling'].items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') * factor
        
        # Source Tracking
        df['DatasetSource'] = meta['source_name']
        
        if meta['source_name'] == "Student_Depression":
            print(f"DEBUG {meta['source_name']}: Columns: {df.columns.tolist()[:15]}")
            if 'SleepHours' in df.columns:
                print(f"DEBUG {meta['source_name']}: SleepHours Head:\n{df['SleepHours'].head()}")
        
        # Keep only intersection of common cols
        available_cols = [c for c in df.columns if c in common_cols]
        df_final = df[available_cols].copy()
        
        # Ensure all common cols exist (NaN etc.)
        for c in common_cols:
            if c not in df_final.columns:
                df_final[c] = np.nan
                
        final_dfs.append(df_final)
    
    # Concatenate
    full_df = pd.concat(final_dfs, ignore_index=True)
    
    # FORCE Numeric Types on Common Features
    numeric_features = [
        "SleepHours", "ExerciseFreq", "SocialActivity", "OnlineStress", "GPA",
        "FamilySupport", "ScreenTime", "AcademicStress", "DietQuality", "SelfEfficacy",
        "PeerRelationship", "FinancialStress", "SleepQuality", "MoodToday", 
        "EnergyLevel", "MotivationLevel", "FocusLevel", "LonelinessScore",
        "InterestLossScore", "NegativeMoodDays", "SocialIsolationScore",
        "ProductivityDrop", "EmotionalStabilityScore", "SleepTrend",
        "JournalingFrequency", "MeditationFrequency", "AppUsageFrequency",
        "PHQ2", "Age", "RiskCategory", "target_raw"
    ]
    
    for col in numeric_features:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            
    # Standardize Gender (0: Male, 1: Female)
    if 'Gender' in full_df.columns:
        full_df['Gender'] = full_df['Gender'].astype(str).str.lower().str.strip().map({
            'male': 0, 'm': 0, '1': 1, '0': 0, 'female': 1, 'f': 1, 'nan': np.nan, 'none': np.nan
        }).fillna(0)
    
    # Final Sparsity Audit
    missing_pct = full_df.isnull().sum() / len(full_df)
    print("\n📊 Final Feature Sparsity Audit:")
    for col in ['SleepHours', 'GPA', 'AcademicStress', 'FinancialStress', 'PHQ2']:
        if col in missing_pct:
            print(f"- {col}: {missing_pct[col]*100:.1f}% missing")
            
    # DROP columns with > 80% missing data
    # EXCEPTION: keep core target and features if possible
    drop_cols = [c for c in missing_pct[missing_pct > 0.8].index if c not in ['RiskCategory', 'DatasetSource', 'SleepHours', 'GPA', 'PHQ2']]
    print(f"🗑️ Dropping sparse features (>80% missing): {drop_cols}")
    full_df = full_df.drop(columns=drop_cols)
    
    # ADD MISSINGNESS INDICATORS
    cols_to_flag = ['SleepHours', 'GPA', 'AcademicStress', 'FinancialStress', 'PHQ2']
    for col in cols_to_flag:
        if col in full_df.columns:
            full_df[f'{col}_missing'] = full_df[col].isna().astype(int)
    
    return full_df

def save_processed_data(df, output_path='data/merged/merged_depression_data.csv'):
    # Ensure BASE_DIR is mind_nova_ai
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(BASE_DIR, output_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    df.to_csv(target_path, index=False)
    print(f"✅ Merged dataset saved to {target_path} (Rows: {len(df)})")

if __name__ == "__main__":
    merged_df = merge_datasets()
    save_processed_data(merged_df)
