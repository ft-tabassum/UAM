import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    # Load data
    data = pd.read_csv("D:/Files_D/Study/Thesis/new_data/aft_2ndversion_lighter.csv")
    
    # Print initial data info
    print("Initial data shape:", data.shape)
    print("Initial columns:", data.columns.tolist())
    print("\nData types:")
    print(data.dtypes.value_counts())
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing_counts = data.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Check original CHOICE values
    print(f"\nOriginal CHOICE values:")
    print(data['CHOICE'].value_counts().sort_index())
    print(f"CHOICE data type: {data['CHOICE'].dtype}")
    print(f"Sample CHOICE values: {data['CHOICE'].head(10).tolist()}")
    
    # Drop unnecessary columns
    cols_to_drop = ["sys_RespNum", "missingvalue"]
    data = data.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Map choice values
    print(f"\nMapping CHOICE values...")
    choice_mapping = {
        "1": 0,  # car
        "2": 1,  # public transport  
        "3": 2,  # autonomous flying taxi
        1: 0,    # handle numeric format
        2: 1,
        3: 2
    }
    
    # Check what values we actually have
    unique_choices = data['CHOICE'].unique()
    print(f"Unique CHOICE values found: {unique_choices}")
    
    # Apply mapping
    data['CHOICE'] = data['CHOICE'].map(choice_mapping).fillna(0).astype(int)
    
    # Verify mapping worked
    print(f"\nAfter mapping CHOICE values:")
    print(data['CHOICE'].value_counts().sort_index())
    
    # Define column types
    numerical_cols = [col for col in data.columns 
                     if data[col].dtype in ['int64', 'float64'] and col != 'CHOICE']
    
    categorical_cols = [col for col in data.columns 
                       if data[col].dtype == 'object' and col != 'CHOICE']
    
    print(f"\nColumn types identified:")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Handle missing values
    print(f"\nHandling missing values...")
    
    # For numerical: Fill with median
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} missing values with median: {median_val}")
    
    # For categorical: Fill with mode
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
            data[col].fillna(mode_val, inplace=True)
            print(f"  Filled {col} missing values with mode: {mode_val}")
    
    # Separate features and target
    X = data.drop(columns=['CHOICE'])
    y = data['CHOICE']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Create preprocessing pipeline - NO STANDARDIZATION
    print(f"\nCreating preprocessing pipeline (no standardization)...")
    
    # Define transformers - only for categorical data
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Create column transformer - only process categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Keep numerical columns as-is
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = []
    feature_names.extend(numerical_cols)  # Numerical features keep original names
    
    # Add categorical feature names
    if categorical_cols:
        categorical_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = categorical_encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
    
    print(f"Processed feature matrix shape: {X_processed.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    
    # Create processed DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    X_processed_df['CHOICE'] = y.values
    
    # Print summary
    print(f"\nFinal processed data shape: {X_processed_df.shape}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10
    print(f"\nFirst few rows:")
    print(X_processed_df.head())
    
    # Verify no missing values
    print(f"\nMissing values in processed data:")
    print(X_processed_df.isnull().sum().sum())
    
    # Verify target distribution
    print(f"\nTarget distribution in processed data:")
    print(X_processed_df['CHOICE'].value_counts().sort_index())
    
    # Save processed data
    output_path = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/LighterModel/LighterModelProcessing_aft.csv"
    X_processed_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    return X_processed_df

if __name__ == "__main__":
    processed_data = main()
