import pandas as pd
import numpy as np
import os

def engineer_deterioration_targets():
    print("🎯 Initializing Deterioration Target Engineering...")
    path = 'data/merged/deterioration_timelines_raw.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    
    # 1. CALCULATE COMPOSITE RISK SCORE (0-10)
    # Different sources provide different indicators. We fill with medians for the composite.
    cols_to_avg = ['Depression', 'Stress', 'Workload'] # Primary indicators
    df['CompositeRisk'] = df[cols_to_avg].mean(axis=1).fillna(5.0) # Baseline if missing
    
    # Identify severe states
    df['IsSevere'] = (df['CompositeRisk'] >= 8.0).astype(int)
    
    # 2. SEQUENCE PROCESSING
    print("📈 Calculating Velocity & Escalation Patterns...")
    processed_sequences = []
    
    for uid, group in df.groupby('user_id'):
        group = group.sort_values('timepoint')
        
        # Calculate Rolling 7-day properties
        group['RollingMean_7D'] = group['CompositeRisk'].rolling(window=7, min_periods=1).mean()
        
        # Velocity = (Current - Prev) / Prev. We'll use day-over-day or window-over-window
        # Here, let's use % change of rolling mean vs previous day rolling mean
        group['RiskVelocity'] = group['RollingMean_7D'].pct_change().fillna(0) * 100
        
        # Counter for consecutive severe days
        group['ConsecutiveSevere'] = group['IsSevere'].groupby((group['IsSevere'] != group['IsSevere'].shift()).cumsum()).cumsum()
        
        # 3. ASSIGN TARGET CLASSES
        # 0 = Stable: Velocity < 5%
        # 1 = Mild: 5-15%
        # 2 = Significant: 15-30%
        # 3 = Crisis: >30% OR 3+ consecutive severe days
        
        def assign_label(row):
            vel = row['RiskVelocity']
            consec = row['ConsecutiveSevere']
            
            if vel > 30 or (consec >= 3):
                return 3 # Crisis
            if vel > 15:
                return 2 # Significant
            if vel > 5:
                return 1 # Mild
            return 0 # Stable
            
        group['Target_Current'] = group.apply(assign_label, axis=1)
        
        # 4. LOOK-AHEAD SHIFT (7-Day Lead)
        # We want to predict Target(t+7) using features(t).
        group['Target'] = group['Target_Current'].shift(-7)
        
        # Drop the last 7 timepoints since we don't have future ground truth for them
        group = group.dropna(subset=['Target'])
        processed_sequences.append(group)
    
    df_targeted = pd.concat(processed_sequences)
    
    print(f"✅ Target Engineering complete (with 7-Day Lead). Rows: {len(df_targeted)}")
    print("Target Distribution (Look-Ahead):")
    print(df_targeted['Target'].value_counts(normalize=True))
    
    os.makedirs('data/processed', exist_ok=True)
    df_targeted.to_csv('data/processed/deterioration_targets.csv', index=False)
    print("💾 Saved to data/processed/deterioration_targets.csv")

if __name__ == "__main__":
    engineer_deterioration_targets()
