import pandas as pd
import numpy as np
import os

def feature_engineering_stress():
    print("🎨 Initializing Stress Feature Engineering & Proxy Simulation...")
    path = 'data/processed/stress_hardened_pool.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path, low_memory=False)
    
    # 1. CORE NUMERIC CLEANING
    num_cols = ['WorkHours', 'SleepHours', 'ScreenTime', 'WorkStress', 'Age', 'ExperienceYears', 'DASS_Stress', 'AcademicStress', 'EmotionalExhaustion']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Normalize WorkStress to 0-10 scale
    df.loc[df['WorkStress'] > 10, 'WorkStress'] = df['WorkStress'] / 10.0
    
    # 2. RECOVERY: MINI-FEATURES & DENOISING (LEAKAGE PURGE)
    print("🚿 Implementing Aggressively Denoised 'Mini' Features...")
    ws_raw = df['WorkStress'].fillna(5.0)
    ee_raw = df['EmotionalExhaustion'].fillna(5.0)
    
    def map_mini_aggressive(val):
        if val < 5.0: return 2.5
        if val < 8.0: return 6.0
        return 9.0
    
    df['WorkStressMini'] = ws_raw.apply(map_mini_aggressive)
    np.random.seed(42)
    df['WorkStressMini'] += np.random.normal(0, 1.0, size=len(df)) # High noise to break 7.5 boundary
    df['WorkStressMini'] = df['WorkStressMini'].clip(0, 10)
    
    df['ExhaustionMini'] = (ee_raw > 6).astype(int)

    # 3. DERIVED BEHAVIORAL FEATURES (Using Mini-Features)
    print("🧬 Calculating Derived Metrics using Mini-Signals...")
    df['AcademicStress'] = df['AcademicStress'].fillna(0)
    # Use Mini for Load
    df['StressLoad'] = (df['WorkStressMini'] + df['AcademicStress']) / 2.0
    
    # Burnout Proxy (Behavioral only)
    work_h = df['WorkHours'].fillna(8)
    slp_h = df['SleepHours'].fillna(7)
    df['BurnoutRiskCalc'] = (work_h + (df['ExhaustionMini'] * 5) - slp_h).clip(0, 20)
    
    # Recovery Gap
    df['RecoveryScore'] = (df.get('JobSatisfaction', 5).fillna(5) + slp_h) / 2.0
    df['RecoveryFailureScore'] = (df['WorkStressMini'] - df['RecoveryScore']).clip(0, 10)
    
    # 4. STOCHASTIC PROXY TRENDS
    print("📊 Simulating Stochastic Trend Proxies...")
    # Use Mini for Spike/Trend
    df['WeeklyStressTrend'] = df['WorkStressMini'] * np.random.uniform(0.8, 1.2, size=len(df))
    df['RecentStressSpike'] = df['WorkStressMini'] + (8 - slp_h).clip(0, 8)
    
    poor_sleep_prob = (df['WorkStressMini'] / 10.0).clip(0, 1)
    df['ConsecutivePoorSleepDays'] = (np.random.poisson(poor_sleep_prob * 3)).astype(int)
    
    workload_prob = (work_h / 12.0).clip(0, 1)
    df['HighWorkloadFrequency'] = (np.random.poisson(workload_prob * 4)).astype(int)

    # 5. MISSINGNESS INDICATORS
    print("🔍 Generating Missingness Indicators...")
    cols_to_trace = ['WorkHours', 'SleepHours', 'ScreenTime', 'WorkStress', 'EmotionalExhaustion']
    for col in cols_to_trace:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # 6. FINAL PURGE: REMOVE ALL CLINICAL SOURCES
    print("🚿 Removing Clinical 'Leaks' and Raw Target Inputs...")
    cols_to_drop = [
        'DASS_Stress', 'burnout_level', 'burnout_risk', 'StressLevel', 'BurnoutRisk',
        'WorkStress', 'EmotionalExhaustion', 'priority'
    ]
    df_inference = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Save
    output_path = 'data/processed/stressed_engineered.csv'
    df_inference.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering complete. Saved to {output_path}")

if __name__ == "__main__":
    feature_engineering_stress()
