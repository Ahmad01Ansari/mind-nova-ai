import pandas as pd
import numpy as np
import os

def engineer_deterioration_features():
    print("🎨 Initializing Final Deterioration Feature Sweep (Target: 80% Recall)...")
    path = 'data/processed/deterioration_targets.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    
    print(f"🧬 Calculating Final Anchor Triad and EscalationVelocity for {len(df['user_id'].unique())} users...")
    processed_features = []
    
    np.random.seed(42)
    
    for uid, group in df.groupby('user_id'):
        group = group.sort_values('timepoint')
        
        # --- PHASE 2: Sharper MoodMini (10% Noise) ---
        anchor_val = group['CompositeRisk'].fillna(5.0)
        def map_mood(val):
            if val < 4.0: return 2.5
            if val < 7.5: return 5.0
            return 8.5
        group['MoodMini'] = anchor_val.apply(map_mood) + np.random.normal(0, 0.5, size=len(group)) # std 0.5 = 10% on 10pt scale
        
        # --- Biological & Environmental Anchors ---
        slp = group['Sleep'].fillna(7.0)
        def map_sleep(val):
            if val < 6.0: return 4.0
            if val < 8.0: return 7.0
            return 9.0
        group['SleepMini'] = slp.apply(map_sleep) + np.random.normal(0, 0.5, size=len(group))
        
        wrk = group['Workload'].fillna(5.0)
        def map_work(val):
            if val < 4.0: return 2.0
            if val < 8.0: return 6.0
            return 9.0
        group['WorkloadMini'] = wrk.apply(map_work) + np.random.normal(0, 0.5, size=len(group))
        
        # --- PHASE 3: EscalationVelocity_Mini ---
        # Detects if multiple signals are moving in the wrong direction simultaneously
        mood_v = group['MoodMini'].diff(3).fillna(0)
        sleep_v = group['SleepMini'].diff(3).fillna(0) * -1 # Sleep decline is positive escalation
        work_v = group['WorkloadMini'].diff(3).fillna(0)
        
        group['EscalationVelocity_Mini'] = (mood_v + sleep_v + work_v).clip(-10, 10)
        
        # --- Hardened Trend Features ---
        slp_f = group['Sleep'].fillna(7.0)
        group['SleepDecline_7D'] = slp_f.rolling(window=7).apply(lambda x: x.iloc[0] - x.iloc[-1] if len(x) > 0 else 0).fillna(0)
        
        risk_threshold = 7.0
        is_risky = (group['CompositeRisk'] > risk_threshold).astype(int)
        group['ConsecutiveRiskDays'] = is_risky.groupby((is_risky != is_risky.shift()).cumsum()).cumsum()
        
        group['Recovery'] = group['Sleep'].fillna(5.0) 
        group['Deficit'] = (group['CompositeRisk'] - group['Recovery']).clip(0, 10)
        group['RecoveryDeficitSlope'] = group['Deficit'].diff().fillna(0)
        
        group['MoodVolatility'] = group['CompositeRisk'].rolling(window=7).std().fillna(0)
        
        vel = group['CompositeRisk'].diff().fillna(0)
        group['BurnoutAcceleration'] = vel.diff().fillna(0)
        
        processed_features.append(group)
        
    df_engineered = pd.concat(processed_features)
    
    # Clean and Clip
    mini_cols = ['MoodMini', 'SleepMini', 'WorkloadMini', 'EscalationVelocity_Mini']
    for c in mini_cols:
        df_engineered[c] = df_engineered[c].clip(-10, 10 if 'Velocity' in c else 10)
    
    inference_cols = [
        'user_id', 'timepoint', 'Source', 'Split', 'Target',
        'MoodMini', 'SleepMini', 'WorkloadMini', 'EscalationVelocity_Mini',
        'SleepDecline_7D', 'ConsecutiveRiskDays',
        'RecoveryDeficitSlope', 'MoodVolatility', 'BurnoutAcceleration'
    ]
    
    df_final = df_engineered[inference_cols]
    
    print(f"✅ Final Feature Engineering complete. Total Features: {len(df_final.columns)}")
    df_final.to_csv('data/processed/deterioration_features_final.csv', index=False)

if __name__ == "__main__":
    engineer_deterioration_features()
