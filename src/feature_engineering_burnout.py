import pandas as pd
import numpy as np
import os

def engineer_burnout_features(input_path='data/merged/merged_burnout_data.csv'):
    print("✨ Engineering Burnout Behavioral Indices (Dual-Model Suite)...")
    df = pd.read_csv(input_path, low_memory=False)
    
    # Create a calculation copy
    calc_df = df.copy()
    
    # 1. Digital Overload Index (Slack, Meetings, Email Sentiment)
    # Scale: 0 to 1
    # Higher = Higher Risk
    if 'SlackActivity' in calc_df.columns:
        slack = calc_df['SlackActivity'].fillna(0)
        meetings = calc_df['MeetingParticipation'].fillna(0)
        sentiment_risk = 1 - calc_df['EmailSentiment'].fillna(0.7)
        
        calc_df['DigitalOverloadIndex'] = (slack + meetings + sentiment_risk) / 3
    else:
        calc_df['DigitalOverloadIndex'] = 0.0
        
    # 2. Workload Intensity
    if 'WorkloadScore' in calc_df.columns:
        overtime_factor = calc_df['OvertimeHours'].fillna(0) / 40
        calc_df['WorkloadIntensity'] = (calc_df['WorkloadScore'].fillna(0.5) + overtime_factor) / 2
    else:
        calc_df['WorkloadIntensity'] = 0.0

    # 3. Traditional Psychiatric Indices (StressLoad)
    stress = calc_df['StressLevel'].fillna(5.0)
    # These might be missing after dropping Student datasets
    academic = calc_df.get('AcademicStress', pd.Series(0, index=df.index)).fillna(0)
    financial = calc_df.get('FinancialStress', pd.Series(0, index=df.index)).fillna(0)
    calc_df['StressLoad'] = (stress + academic + financial) / 30 
    
    # 4. Sleep & Recovery Indices
    sleep_h = calc_df['SleepHours'].fillna(7.0)
    calc_df['SleepDebt'] = np.maximum(0, (8 - sleep_h) / 8)
    
    # Use .get() for PhysicalActivity and SleepQuality
    recovery_phys = calc_df.get('PhysicalActivity', pd.Series(1.0, index=df.index)).fillna(1.0) 
    recovery_sleep = calc_df.get('SleepQuality', pd.Series(5.0, index=df.index)).fillna(5.0)
    calc_df['RecoveryScore'] = (recovery_phys / 10) + (recovery_sleep / 10)
    calc_df['RecoveryScore'] = calc_df['RecoveryScore'] / 2
    
    # 5. Missingness Indicators (CRITICAL for Dual Routing)
    missing_flags = ['OvertimeHours', 'SlackActivity', 'MeetingParticipation', 'EmailSentiment', 'WorkloadScore']
    for col in missing_flags:
        if col in df.columns:
            calc_df[f'{col}_missing'] = df[col].isnull().astype(int)
        else:
            calc_df[f'{col}_missing'] = 1

    # Extra missingness for self-report features
    for col in ['ExperienceYears', 'WorkHours', 'GPA']:
        if col in df.columns:
            calc_df[f'{col}_missing'] = df[col].isnull().astype(int)

    # Save
    os.makedirs('data/processed', exist_ok=True)
    calc_df.to_csv('data/processed/processed_burnout_data.csv', index=False)
    
    print(f"✅ Feature engineering complete. Total Rows: {len(calc_df)} | Features: {len(calc_df.columns)}")
    return calc_df

if __name__ == "__main__":
    engineer_burnout_features()
