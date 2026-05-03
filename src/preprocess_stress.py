import pandas as pd
import numpy as np
import os
import glob

def preprocess_stress():
    print("🚀 Initializing Multi-Source Stress Data Preprocessing...")
    os.makedirs('data/merged', exist_ok=True)
    
    merged_data = []
    
    # --- HELPER: Moderate Composite Target Logic ---
    def assign_target(df):
        # Default to 0
        df['Target'] = 0
        
        # Ensure Numeric and Series-shaped for comparisons
        ws_num = pd.to_numeric(df['WorkStress'], errors='coerce') if 'WorkStress' in df.columns else pd.Series(np.nan, index=df.index)
        fe_num = pd.to_numeric(df['EmotionalExhaustion'], errors='coerce') if 'EmotionalExhaustion' in df.columns else pd.Series(np.nan, index=df.index)
        dass_num = pd.to_numeric(df['DASS_Stress'], errors='coerce') if 'DASS_Stress' in df.columns else pd.Series(np.nan, index=df.index)
        pss_num = pd.to_numeric(df['PSS_10'], errors='coerce') if 'PSS_10' in df.columns else pd.Series(np.nan, index=df.index)
        
        # 1. DASS_Stress >= 15 (Moderate/Severe)
        df.loc[dass_num >= 15, 'Target'] = 1
            
        # 2. PSS_10 >= 20
        df.loc[pss_num >= 20, 'Target'] = 1
            
        # 3. Numeric Stress Level >= 7.5
        df.loc[ws_num >= 7.5, 'Target'] = 1
                
        # 4. Composite: Workload > 8 AND Sleep < 5 AND Fatigue > 7
        workload_col = 'WorkHours' if 'WorkHours' in df.columns else None
        sleep_col = 'SleepHours' if 'SleepHours' in df.columns else None
        
        if workload_col and sleep_col:
             mask = (df[workload_col] > 8) & (df[sleep_col] < 5) & (fe_num > 7)
             df.loc[mask, 'Target'] = 1
             
        # 5. Composite: EmotionalExhaustion > 7 AND WorkStress > 7
        mask = (ws_num > 7) & (fe_num > 7)
        df.loc[mask, 'Target'] = 1
                
        return df

    # --- SOURCE 1: Healthcare Workforce ---
    print("📥 Loading Healthcare Workforce...")
    hc_path = 'data/raw/Healthcare Workforce Mental Health Dataset.csv'
    if os.path.exists(hc_path):
        hc = pd.read_csv(hc_path)
        # Stress Level 1-10? Check audit: Mean 6.5, Max 9.0. Looks fine.
        hc_mapped = pd.DataFrame({
            'WorkStress': hc['Stress Level'],
            'EmotionalExhaustion': hc['Burnout Frequency'],
            'JobSatisfaction': hc['Job Satisfaction'],
            'DatasetSource': 'Healthcare_Workforce'
        })
        freq_map = {'Never': 1, 'Rarely': 3, 'Sometimes': 5, 'Frequently': 8, 'Always': 10}
        hc_mapped['EmotionalExhaustion'] = hc_mapped['EmotionalExhaustion'].map(freq_map).fillna(5)
        
        hc_mapped = assign_target(hc_mapped)
        merged_data.append(hc_mapped)

    # --- SOURCE 2: Remote Work (8000 Rows) ---
    print("📥 Loading Remote Work...")
    remote_path = 'data/raw/Remote_Work_Productivity_And_Burnout_8000_Rows.csv'
    if os.path.exists(remote_path):
        rw = pd.read_csv(remote_path)
        # MAP CATEGORICAL STRESS TO NUMERIC 0-10
        stress_map = {'High': 8.5, 'Medium': 5.0, 'Low': 2.5}
        
        rw_mapped = pd.DataFrame({
            'WorkHours': rw['daily_work_hours'],
            'SleepHours': rw['sleep_hours'],
            'ScreenTime': rw['daily_screen_time_hours'],
            'WorkStress': rw['stress_level'].map(stress_map),
            'DatasetSource': 'Remote_Work'
        })
        b_map = {'Low': 2.5, 'Medium': 5.0, 'High': 8.5}
        rw_mapped['EmotionalExhaustion'] = rw['burnout_level'].map(b_map).fillna(5.0)
        
        rw_mapped = assign_target(rw_mapped)
        merged_data.append(rw_mapped)

    # --- SOURCE 3: Corporate Stress (Numbered) ---
    print("📥 Loading Corporate Stress (Numbered)...")
    corp_path = 'data/raw/corporate_stress_dataset.csv'
    if os.path.exists(corp_path):
        corp = pd.read_csv(corp_path, header=None)
        corp_mapped = pd.DataFrame({
            'Age': pd.to_numeric(corp[1], errors='coerce'),
            'ExperienceYears': pd.to_numeric(corp[5], errors='coerce'),
            'WorkStress': pd.to_numeric(corp[14], errors='coerce'),
            'DatasetSource': 'Corporate_Stress'
        })
        corp_mapped = assign_target(corp_mapped)
        merged_data.append(corp_mapped)

    # --- SOURCE 4: DASS-42 (data.csv) ---
    print("📥 Loading DASS-42 Data...")
    dass_path = 'data/raw/data.csv'
    if os.path.exists(dass_path):
        dass = pd.read_csv(dass_path, sep='\t')
        stress_items = ['Q1A', 'Q6A', 'Q8A', 'Q11A', 'Q12A', 'Q14A', 'Q18A', 'Q22A', 'Q27A', 'Q29A', 'Q32A', 'Q33A', 'Q35A', 'Q39A']
        dass['DASS_Stress'] = dass[stress_items].apply(lambda x: x - 1).sum(axis=1)
        
        dass_mapped = pd.DataFrame({
            'Age': dass['age'],
            'DASS_Stress': dass['DASS_Stress'],
            'DatasetSource': 'DASS_Clinical'
        })
        dass_mapped = assign_target(dass_mapped)
        merged_data.append(dass_mapped)

    # --- SOURCE 5: Student Stress Monitoring ---
    print("📥 Loading Student Stress...")
    st_path = 'data/raw/Student Stress Monitoring Datasets_StressLevelDataset.csv'
    if os.path.exists(st_path):
        st = pd.read_csv(st_path)
        # MAP [0, 1, 2] -> [2.5, 5.0, 8.5]
        student_stress_map = {0: 2.5, 1: 5.0, 2: 8.5}
        
        st_mapped = pd.DataFrame({
            'WorkStress': st['stress_level'].map(student_stress_map),
            'AcademicStress': st['study_load'],
            'DatasetSource': 'Student_Stress_Mon'
        })
        st_mapped = assign_target(st_mapped)
        merged_data.append(st_mapped)

    # 5. MERGE & CLEAN
    print("🧩 Merging all cohorts...")
    final_df = pd.concat(merged_data, ignore_index=True)
    
    # Handle Global Weighting in Target (as metadata for now)
    # We will use sample_weight in training
    
    # Save Merged Data
    output_path = 'data/merged/merged_stress_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing complete. Total records: {len(final_df)}")
    print(f"   Saved to {output_path}")

if __name__ == "__main__":
    preprocess_stress()
