import pandas as pd
import numpy as np

def engineer_depression_features(df):
    """
    Creates derived features for Depression Risk Prediction.
    Adjusted based on actually available features in the merged dataset.
    """
    df = df.copy()
    
    # Fill NAs with medians for calculation
    # Only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 1. StressLoad (Academic + Financial)
    df['StressLoad'] = df['AcademicStress'] + df['FinancialStress']
    
    # 2. BurnoutRisk (Stress - Rest)
    # Using SleepHours as the rest indicator
    df['BurnoutRisk'] = df['AcademicStress'] - df['SleepHours']
    
    # 3. RiskIndex 
    df['RiskIndex'] = df['StressLoad'] - df['SleepHours']
    
    # 4. AcademicImpact
    # Low GPA + High Academic Stress
    df['AcademicImpact'] = df['AcademicStress'] / (df['GPA'] + 1)
    
    # 5. Age-Adjusted Stress
    df['AgeStressRatio'] = df['StressLoad'] / (df['Age'] + 1)
    
    # 6. Target Interaction: PHQ2 to Stress
    df['SymptomStressRatio'] = df['PHQ2'] / (df['StressLoad'] + 1)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/merged/merged_depression_data.csv')
    df_engineered = engineer_depression_features(df)
    df_engineered.to_csv('data/processed/processed_depression_data.csv', index=False)
    print(f"✨ Feature engineering complete. Saved to data/processed/processed_depression_data.csv")
    print(f"Columns: {df_engineered.columns.tolist()}")
