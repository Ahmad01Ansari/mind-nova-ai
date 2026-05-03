import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Step 1: Data Loading
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def basic_inspection(df):
    """
    Step 1: Data Inspection
    """
    print("--- First 5 rows ---")
    print(df.head())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    print("\n--- Target Class Distribution ---")
    print(df['MentalHealthStatus'].value_counts(normalize=True))

def clean_data(df):
    """
    Step 3: Data Cleaning
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_shape = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    print(f"Removed {initial_shape - df_clean.shape[0]} duplicate rows.")
    
    # Handle missing values (if any)
    if df_clean.isnull().sum().any():
        # Using median for numerical columns
        df_clean.fillna(df_clean.median(), inplace=True)
        print("Missing values handled using median imputation.")
    
    # Standardize column names (already fairly standard, but ensuring consistency)
    df_clean.columns = [col.strip() for col in df_clean.columns]
    
    return df_clean

if __name__ == "__main__":
    # Test loading
    dataset_path = "/home/ahmad10raza/Documents/Major Projects/MindNova/mind_nova_ai/data/raw/Univsersiyt_Student_Mental_health_data.csv"
    data = load_data(dataset_path)
    basic_inspection(data)
    data = clean_data(data)
