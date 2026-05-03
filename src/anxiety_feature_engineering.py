import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Step 4: Feature Engineering
    """
    df_feat = df.copy()
    
    # Stress metrics aggregation
    # StressLoad = OnlineStress + AcademicStress + FinancialStress
    df_feat['StressLoad'] = df_feat['OnlineStress'] + df_feat['AcademicStress'] + df_feat['FinancialStress']
    
    # Lifestyle index aggregation
    # LifestyleScore = SleepHours + ExerciseFreq + DietQuality + SleepQuality
    df_feat['LifestyleScore'] = df_feat['SleepHours'] + df_feat['ExerciseFreq'] + df_feat['DietQuality'] + df_feat['SleepQuality']
    
    # Support network aggregation
    # SupportScore = FamilySupport + PeerRelationship + SocialActivity
    df_feat['SupportScore'] = df_feat['FamilySupport'] + (df_feat['PeerRelationship'] if 'PeerRelationship' in df_feat else df_feat['SocialActivity']) + df_feat['SocialActivity']
    
    # Composite Risk score
    # RiskScore = ScreenTime + OnlineStress + FinancialStress
    df_feat['RiskScore'] = (df_feat['ScreenTime'] if 'ScreenTime' in df_feat else 0) + df_feat['OnlineStress'] + df_feat['FinancialStress']
    
    # Personal Wellness index
    # WellnessScore = SelfEfficacy + DietQuality + ExerciseFreq
    df_feat['WellnessScore'] = df_feat['SelfEfficacy'] + df_feat['DietQuality'] + df_feat['ExerciseFreq']
    
    return df_feat

def get_feature_versions(df):
    """
    Separates the dataset into Version A (All features) and Version B (No diagnostic leakage)
    """
    target = 'MentalHealthStatus'
    
    # Version A: All features
    X_A = df.drop(columns=[target])
    y_A = df[target]
    
    # Version B: Remove GAD7 and PHQ9 (and potential leakage columns)
    diag_columns = ['GAD7', 'PHQ9']
    existing_diag = [col for col in diag_columns if col in df.columns]
    
    X_B = X_A.drop(columns=existing_diag)
    y_B = y_A
    
    return (X_A, y_A), (X_B, y_B)

def scale_features(X):
    """
    Standardizes features to zero mean and unit variance
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler
