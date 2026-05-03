import pandas as pd
import numpy as np

def audit_burnout_split(path='data/merged/merged_burnout_data.csv'):
    print("🔍 Auditing Burnout Feature Split...")
    df = pd.read_csv(path)
    
    # Behavioral columns
    behavioral_cols = ['SlackActivity', 'MeetingParticipation', 'EmailSentiment', 'WorkloadScore']
    
    audit_results = []
    for source in df['DatasetSource'].unique():
        sub = df[df['DatasetSource'] == source]
        # Check percentage of rows with ANY behavioral data
        coverage = sub[behavioral_cols].notnull().any(axis=1).mean() * 100
        audit_results.append({
            'Source': source,
            'Records': len(sub),
            'Behavioral_Coverage_%': f"{coverage:.1f}%",
            'Target_Rate_%': f"{sub['RiskCategory'].mean():.1%}"
        })
    
    results_df = pd.DataFrame(audit_results)
    print(results_df.to_string(index=False))
    
    # Check overlapping features
    all_cols = df.columns
    print(f"\nTotal Columns: {list(all_cols)}")
    
if __name__ == "__main__":
    audit_burnout_split()
