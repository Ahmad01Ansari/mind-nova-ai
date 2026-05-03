import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_hybrid_features(df):
    """
    Step 2: Advanced Feature Engineering
    Implementation of the 10 requested lifestyle/behavioral metrics.
    """
    df = df.copy()
    
    # 1. StressLoad = OnlineStress + AcademicStress + FinancialStress
    df['StressLoad'] = df['OnlineStress'] + df['AcademicStress'] + df['FinancialStress']
    
    # 2. LifestyleScore = SleepHours + ExerciseFreq + DietQuality + SleepQuality
    df['LifestyleScore'] = df['SleepHours'] + df['ExerciseFreq'] + df['DietQuality'] + df['SleepQuality']
    
    # 3. SupportScore = FamilySupport + PeerRelationship + SocialActivity
    df['SupportScore'] = df['FamilySupport'] + df['PeerRelationship'] + df['SocialActivity']
    
    # 4. DigitalOverload = ScreenTime + OnlineStress
    df['DigitalOverload'] = df['ScreenTime'] + df['OnlineStress']
    
    # 5. MentalResilience = SelfEfficacy + FamilySupport + PeerRelationship
    df['MentalResilience'] = df['SelfEfficacy'] + df['FamilySupport'] + df['PeerRelationship']
    
    # 6. BurnoutRisk = AcademicStress + ScreenTime - SleepHours
    df['BurnoutRisk'] = df['AcademicStress'] + df['ScreenTime'] - df['SleepHours']
    
    # 7. RiskIndex = StressLoad + ScreenTime - LifestyleScore
    df['RiskIndex'] = df['StressLoad'] + df['ScreenTime'] - df['LifestyleScore']
    
    return df

def scale_features(df, feature_cols):
    """
    Standardize the feature set for optimal model performance.
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

if __name__ == "__main__":
    from hybrid_preprocess import load_data, create_hybrid_labels
    
    PATH = 'data/raw/Univsersiyt_Student_Mental_health_data.csv'
    df = create_hybrid_labels(load_data(PATH))
    df_feat = engineer_hybrid_features(df)
    
    print("New Engineered Features:")
    new_cols = ['StressLoad', 'LifestyleScore', 'SupportScore', 'DigitalOverload', 
                'MentalResilience', 'BurnoutRisk', 'WellbeingIndex', 'RiskIndex']
    print(df_feat[new_cols].head())
    print("\nFeature count:", len(df_feat.columns))
