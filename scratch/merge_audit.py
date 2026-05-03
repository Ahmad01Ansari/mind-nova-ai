import pandas as pd
import os

def audit_merge_sparsity():
    RAW_DIR = '../data/raw/' # Adjust to workspace root
    # Wait, in the terminal I'm in mind_nova_ai
    # Actually, I'll just load them directly
    
    files = [
        'Univsersiyt_Student_Mental_health_data.csv',
        'Student Depression Dataset.csv',
        'Global_Mental_Health_Dataset_2025.csv',
        'Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv',
        'data.csv'
    ]
    
    project_root = '/home/ahmad10raza/Documents/Major Projects/MindNova/mind_nova_ai'
    raw_path = os.path.join(os.path.dirname(project_root), 'data/raw')
    
    for f in files:
        path = os.path.join(raw_path, f)
        if not os.path.exists(path):
            print(f"❌ {f} not found at {path}")
            continue
        
        sep = '\t' if f == 'data.csv' else ','
        df = pd.read_csv(path, sep=sep, nrows=5)
        print(f"--- {f} ---")
        print(f"Columns: {df.columns.tolist()}")
        print("-" * 30)

if __name__ == "__main__":
    audit_merge_sparsity()
