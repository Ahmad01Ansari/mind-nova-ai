import pandas as pd
import numpy as np
import os

def de_duplicate_stress():
    print("🧹 Initializing Global Priority-Based De-duplication...")
    path = 'data/merged/merged_stress_data.csv'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path, low_memory=False)
    initial_count = len(df)
    
    # 1. DEFINE SOURCE PRIORITY
    # Healthcare Workforce > Remote Work > Corporate Stress > Student Stress > DASS Clinical
    source_priority = {
        'Healthcare_Workforce': 1,
        'Remote_Work': 2,
        'Corporate_Stress': 3,
        'Student_Stress_Mon': 4,
        'DASS_Clinical': 5
    }
    
    # Add priority column for sorting
    df['priority'] = df['DatasetSource'].map(source_priority).fillna(99)
    
    # Sort by priority (lower number = higher priority)
    df = df.sort_values(by='priority')
    
    # 2. DEFINE BEHAVIORAL PROFILE FOR DUPLICATE CHECK
    # Identify non-id/non-source columns that represent the behavioral fingerprint
    behavioral_cols = [c for c in df.columns if c not in ['DatasetSource', 'Target', 'priority', 'user_id', 'Employee_Id', 'id', 'Employee_Id.1']]
    
    # 3. DROP DUPLICATES
    # 'keep=first' will preserve the row from the highest priority source due to our sort
    df_hardened = df.drop_duplicates(subset=behavioral_cols, keep='first')
    
    final_count = len(df_hardened)
    dropped_count = initial_count - final_count
    
    print(f"📊 De-duplication Summary:")
    print(f"   Initial Rows: {initial_count}")
    print(f"   Dropped Rows: {dropped_count} ({dropped_count/initial_count:.2%})")
    print(f"   Hardened Rows: {final_count}")
    
    # 4. EXPORT
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/stress_hardened_pool.csv'
    df_hardened.drop(columns=['priority']).to_csv(output_path, index=False)
    print(f"✅ Data Hardening complete. Saved to {output_path}")

if __name__ == "__main__":
    de_duplicate_stress()
