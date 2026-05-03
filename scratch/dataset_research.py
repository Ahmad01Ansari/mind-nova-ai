import pandas as pd
import numpy as np
import os

files = [
    'data/raw/Univsersiyt_Student_Mental_health_data.csv',
    'data/raw/Student Depression Dataset.csv',
    'data/raw/Global_Mental_Health_Dataset_2025.csv',
    'data/raw/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv',
    'data/raw/data.csv'
]

print("🔍 MindNova Depression Model: Dataset Research Audit\n")

for f in files:
    if not os.path.exists(f):
        print(f"❌ Missing: {f}")
        continue
    
    try:
        df = pd.read_csv(f)
        print(f"--- Dataset: {f} ---")
        print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
        print(f"Columns: {df.columns.tolist()[:10]}...") # Print first 10
        
        # Check potential targets
        potential_targets = [c for c in df.columns if 'phq' in c.lower() or 'depression' in c.lower()]
        print(f"Potential Targets: {potential_targets}")
        
        # Check potential features
        potential_features = [c for c in df.columns if any(p in c.lower() for p in ['sleep', 'stress', 'gpa', 'exercise', 'activity', 'mood'])]
        print(f"Potential Features: {potential_features}")
        
        # Basic stats for a few columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print(f"Numeric Ranges (first 3): \n{df[numeric_cols[:3]].agg(['min', 'max'])}")
        
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"🔥 Error reading {f}: {e}")
