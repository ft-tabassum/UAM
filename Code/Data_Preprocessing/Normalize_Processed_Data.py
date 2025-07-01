import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_processed_data():
    """
    Normalize both reduced model data and processed synthetic data
    for probability calculations.
    """
    
    print("Normalizing processed data for probability calculations...")
    
    # Load both datasets
    print("\nLoading datasets...")
    reduced_data = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/reduced_data_processed.csv')
    synthetic_data = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/synthetic_data_processed.csv')
    
    print(f"Reduced data shape: {reduced_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    # Check column compatibility
    print("\nChecking column compatibility...")
    reduced_cols = set(reduced_data.columns)
    synthetic_cols = set(synthetic_data.columns)
    
    common_cols = reduced_cols.intersection(synthetic_cols)
    reduced_only = reduced_cols - synthetic_cols
    synthetic_only = synthetic_cols - reduced_cols
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Reduced only: {len(reduced_only)}")
    print(f"Synthetic only: {len(synthetic_only)}")
    
    if len(common_cols) < 5:  # At least need mode and a few features
        print("Warning: Very few common columns between datasets!")
        print(f"Common columns: {sorted(common_cols)}")
    
    # Use only common columns for compatibility
    common_cols_list = sorted(list(common_cols))
    print(f"\nUsing common columns: {common_cols_list}")
    
    # Prepare data with common columns
    reduced_common = reduced_data[common_cols_list].copy()
    synthetic_common = synthetic_data[common_cols_list].copy()
    
    # Separate target and features
    target_col = 'mode'
    if target_col in common_cols_list:
        feature_cols = [col for col in common_cols_list if col != target_col]
    else:
        print("Warning: 'mode' column not found in common columns!")
        feature_cols = common_cols_list
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target column: {target_col}")
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col in reduced_common.columns:
            if reduced_common[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Normalize numeric columns
    print("\nNormalizing numeric columns...")
    
    # Choose scaler type
    scaler_type = 'minmax'  # Options: 'minmax', 'standard', 'robust'
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    
    # Fit scaler on reduced data (training data)
    if numeric_cols:
        reduced_numeric = reduced_common[numeric_cols]
        scaler.fit(reduced_numeric)
        
        # Transform both datasets
        reduced_common[numeric_cols] = scaler.transform(reduced_numeric)
        synthetic_common[numeric_cols] = scaler.transform(synthetic_common[numeric_cols])
        
        print(f"Applied {scaler_type} scaling to {len(numeric_cols)} numeric columns")
    
    # Save normalized datasets
    print("\nSaving normalized datasets...")
    
    # Create output directory if it doesn't exist
    output_dir = 'D:/PythonProject/Result/Data_Preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save normalized reduced data
    reduced_output_path = os.path.join(output_dir, 'reduced_data_normalized.csv')
    reduced_common.to_csv(reduced_output_path, index=False)
    print(f"Normalized reduced data saved: {reduced_output_path}")
    
    # Save normalized synthetic data
    synthetic_output_path = os.path.join(output_dir, 'synthetic_data_normalized.csv')
    synthetic_common.to_csv(synthetic_output_path, index=False)
    print(f"Normalized synthetic data saved: {synthetic_output_path}")
    
    # Save the scaler for future use
    if numeric_cols:
        scaler_path = os.path.join(output_dir, 'normalization_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved: {scaler_path}")
    
    # Display statistics
    print("\nDataset statistics after normalization:")
    
    print("\nReduced data statistics:")
    print(f"Shape: {reduced_common.shape}")
    if target_col in reduced_common.columns:
        mode_counts = reduced_common[target_col].value_counts().sort_index()
        print("Mode distribution:")
        for mode, count in mode_counts.items():
            print(f"  Mode {mode}: {count} ({count/len(reduced_common)*100:.1f}%)")
    
    print("\nSynthetic data statistics:")
    print(f"Shape: {synthetic_common.shape}")
    if target_col in synthetic_common.columns:
        mode_counts = synthetic_common[target_col].value_counts().sort_index()
        print("Mode distribution:")
        for mode, count in mode_counts.items():
            print(f"  Mode {mode}: {count} ({count/len(synthetic_common)*100:.1f}%)")
    
    # Check for missing values
    print("\nMissing values check:")
    reduced_missing = reduced_common.isnull().sum().sum()
    synthetic_missing = synthetic_common.isnull().sum().sum()
    print(f"Reduced data missing values: {reduced_missing}")
    print(f"Synthetic data missing values: {synthetic_missing}")
    
    # Feature ranges after normalization
    if numeric_cols:
        print(f"\nNumeric feature ranges after {scaler_type} normalization:")
        for col in numeric_cols[:5]:  # Show first 5 features
            min_val = reduced_common[col].min()
            max_val = reduced_common[col].max()
            print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
    
    print(f"\nNormalization completed successfully!")
    print(f"Both datasets are now ready for probability calculations.")
    
    return reduced_common, synthetic_common, scaler

def create_combined_dataset():
    """
    Create a combined dataset from both normalized datasets for training.
    """
    
    print("\nCreating combined dataset...")
    
    # Load normalized datasets
    reduced_normalized = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/reduced_data_normalized.csv')
    synthetic_normalized = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/synthetic_data_normalized.csv')
    
    # Add source identifier
    reduced_normalized['data_source'] = 'reduced'
    synthetic_normalized['data_source'] = 'synthetic'
    
    # Combine datasets
    combined_data = pd.concat([reduced_normalized, synthetic_normalized], ignore_index=True)
    
    # Save combined dataset
    output_path = 'D:/PythonProject/Result/Data_Preprocessing/combined_normalized_data.csv'
    combined_data.to_csv(output_path, index=False)
    
    print(f"Combined dataset saved: {output_path}")
    print(f"Combined dataset shape: {combined_data.shape}")
    
    # Show distribution
    source_counts = combined_data['data_source'].value_counts()
    print("Data source distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} ({count/len(combined_data)*100:.1f}%)")
    
    return combined_data

if __name__ == "__main__":
    # Normalize both datasets
    reduced_norm, synthetic_norm, scaler = normalize_processed_data()
    
    # Create combined dataset
    combined_data = create_combined_dataset()
    
    print("\nAll processing completed!")
    print("Files created:")
    print("  - reduced_data_normalized.csv")
    print("  - synthetic_data_normalized.csv") 
    print("  - combined_RSnormalized_data.csv")
    print("  - normalization_scaler.joblib") 