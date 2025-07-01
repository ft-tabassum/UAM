import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def process_synthetic_data():
    """
    Process synthetic trip data to match the structure of reduced model data
    using the specific mappings provided by the user.
    """
    
    print("Processing synthetic trip data to match reduced model structure...")
    
    # Load the synthetic trip data
    synthetic_data = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/Synthetic_tripData.csv')
    
    print(f"Original synthetic data shape: {synthetic_data.shape}")
    print(f"Original synthetic data columns: {list(synthetic_data.columns)}")
    
    # Create a copy to work with
    processed_data = synthetic_data.copy()
    
    # 1. Process Target Variable (Mode)
    print("\nProcessing target variable (mode)...")
    mode_mapping = {
        'autoPassenger': 0,
        'autoDriver': 1,
        'bus': 2,
        'train': 3,
        'bicycle': 4,
        'walk': 5
    }
    processed_data['mode'] = processed_data['mode'].map(mode_mapping)
    print(f"Mode mapping applied. Unique values: {processed_data['mode'].unique()}")
    
    # 2. Process Categorical Variables
    print("\nProcessing categorical variables...")
    
    # Gender mapping
    gender_mapping = {1: 'male', 2: 'female'}
    processed_data['gender'] = processed_data['gender'].map(gender_mapping)
    
    # Employment mapping
    employment_mapping = {1: 'Working full-time', 2: 'Unemployed', 3: 'Pupil, student or apprentice/intern'}
    processed_data['employment'] = processed_data['employment'].map(employment_mapping)
    
    # Driving license mapping
    processed_data['driving license'] = processed_data['driving license'].map({True: 'Yes', False: 'No'})
    
    # Purpose mapping
    purpose_mapping = {
        'HBW': 'Home-based work',
        'HBE': 'Home-based education', 
        'HBS': 'Home-based shopping',
        'HBR': 'Home-based recreation',
        'HBO': 'Home-based other',
        'NHBW': 'Non-home-based work',
        'NHBO': 'Non-home-based other'
    }
    processed_data['purpose'] = processed_data['purpose'].map(purpose_mapping)
    
    print("Categorical variables processed.")
    
    # 3. Process Age (convert to age groups)
    print("\nProcessing age...")
    # First, let's see what age values we have
    print(f"Age values in data: {sorted(processed_data['age'].unique())}")
    
    # Since age is already numeric, we'll create age groups
    def categorize_age(age):
        if age < 18:
            return 0  # '0-17'
        elif age < 30:
            return 1  # '18-29'
        elif age < 40:
            return 2  # '30-39'
        elif age < 50:
            return 3  # '40-49'
        elif age < 60:
            return 4  # '50-59'
        elif age < 70:
            return 5  # '60-69'
        elif age < 80:
            return 6  # '70-79'
        else:
            return 7  # 'I prefer not to answer'
    
    processed_data['age'] = processed_data['age'].apply(categorize_age)
    print(f"Age categorized. Unique values: {sorted(processed_data['age'].unique())}")
    
    # 4. Process Income (convert to income groups)
    print("\nProcessing income...")
    # First, let's see what income values we have
    print(f"Income values in data: {sorted(processed_data['income'].unique())}")
    
    # Since income is already numeric, we'll create income groups
    def categorize_income(income):
        if income == 0:
            return 0  # 'I prefer not to answer'
        elif income < 1000:
            return 2  # 'Under € 1000'
        elif income < 2000:
            return 3  # '€ 1000 to less than € 2000'
        elif income < 3000:
            return 4  # '€ 2000 to less than € 3000'
        elif income < 4000:
            return 5  # '€ 3000 to less than € 4000'
        elif income < 5000:
            return 6  # '€ 4000 to less than € 5000'
        elif income < 6000:
            return 7  # '€ 5000 to less than € 6000'
        elif income < 7000:
            return 8  # '€ 6000 to less than € 7000'
        else:
            return 9  # '€ 7000 or more'
    
    processed_data['income'] = processed_data['income'].apply(categorize_income)
    print(f"Income categorized. Unique values: {sorted(processed_data['income'].unique())}")
    
    # 5. Select and rename relevant columns
    print("\nSelecting and organizing columns...")
    
    # Define the columns we want to keep and their new names - matching reduced model exactly
    columns_to_keep = {
        'driving license': 'driving license',
        'gender': 'gender',
        'age': 'age',
        'employment': 'employment',
        'income': 'monthly income',
        'purpose': 'Purpose',
        'household car': 'household car',
        'mode': 'mode'
    }
    
    # Add veh_time using auto time
    processed_data['veh_time'] = processed_data['time_auto']  # Use auto time as vehicle time
    columns_to_keep['veh_time'] = 'veh_time'
    
    # Check which columns exist in the data
    available_columns = []
    missing_columns = []
    
    for old_name, new_name in columns_to_keep.items():
        if old_name in processed_data.columns:
            available_columns.append((old_name, new_name))
        else:
            missing_columns.append(old_name)
    
    print(f"Available columns: {len(available_columns)}")
    print(f"Missing columns: {len(missing_columns)}")
    if missing_columns:
        print(f"Missing: {missing_columns}")
    
    # Select only available columns
    selected_data = processed_data[[col[0] for col in available_columns]].copy()
    
    # Rename columns
    column_mapping = {old: new for old, new in available_columns}
    selected_data = selected_data.rename(columns=column_mapping)
    
    print(f"Selected data shape: {selected_data.shape}")
    print(f"Selected columns: {list(selected_data.columns)}")
    
    # 6. Identify ordinal and nominal columns - matching reduced model exactly
    print("\nIdentifying column types...")
    
    ordinal_cols = ['age', 'monthly income', 'veh_time']
    nominal_cols = ['driving license', 'Purpose', 'household car', 'gender', 'employment']
    
    print(f"Ordinal columns: {ordinal_cols}")
    print(f"Nominal columns: {nominal_cols}")
    
    # 7. Handle missing values
    print("\nHandling missing values...")
    
    # Replace missing nominal with '0' string
    if nominal_cols:
        selected_data[nominal_cols] = selected_data[nominal_cols].fillna('0').astype(str)
    
    # Replace missing ordinal with numeric 0
    for col in ordinal_cols:
        selected_data[col] = selected_data[col].fillna(0)
    
    # 8. One-hot encode nominal variables
    print("\nOne-hot encoding nominal variables...")
    
    if nominal_cols:
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('nominal', categorical_transformer, nominal_cols)
        ], remainder='passthrough')
        
        # Separate features and target
        X = selected_data.drop(columns=['mode'])
        y = selected_data['mode']
        
        X_processed = preprocessor.fit_transform(X)
        
        passthrough_cols = [col for col in X.columns if col not in nominal_cols]
        feature_names = list(preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(
            nominal_cols)) + passthrough_cols
        
        # Convert to dense array if sparse
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        X_processed_df['mode'] = y.values
    else:
        # If no nominal columns, just use the data as is
        X_processed_df = selected_data.copy()
    
    print(f"Final processed data shape: {X_processed_df.shape}")
    print(f"Final columns: {list(X_processed_df.columns)}")
    
    # 9. Save the processed data
    output_path = 'D:/PythonProject/Result/Data_Preprocessing/synthetic_data_processed.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_processed_df.to_csv(output_path, index=False)
    
    print(f"\nProcessing completed successfully!")
    print(f"Output saved as: {output_path}")
    print(f"Final dataset: {len(X_processed_df)} rows, {len(X_processed_df.columns)} columns")
    
    # Display basic statistics
    print("\nBasic statistics of processed dataset:")
    print(X_processed_df.describe())
    
    # Check for missing values
    print("\nMissing values in processed dataset:")
    missing_values = X_processed_df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found")
    
    # Show target distribution
    print(f"\nTarget variable (mode) distribution:")
    mode_counts = X_processed_df['mode'].value_counts().sort_index()
    for mode, count in mode_counts.items():
        print(f"  Mode {mode}: {count} trips ({count/len(X_processed_df)*100:.1f}%)")
    
    # 10. Align features and one-hot encoding with reduced model
    print("\nAligning features and one-hot encoding with reduced model...")
    reduced = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/reduced_model_normalized.csv')
    def standardize(col):
        return col.lower().replace(' ', '').replace('_', '')
    reduced_cols_std = {standardize(col): col for col in reduced.columns}
    synthetic_cols_std = {standardize(col): col for col in X_processed_df.columns}
    cols_to_keep = []
    for std_col, orig_col in reduced_cols_std.items():
        if std_col in synthetic_cols_std:
            cols_to_keep.append(synthetic_cols_std[std_col])
        else:
            # If missing, add a column with zeros
            X_processed_df[orig_col] = 0
            cols_to_keep.append(orig_col)
    # Remove extra columns from synthetic data
    synthetic_aligned = X_processed_df[cols_to_keep]
    # Reorder columns to match reduced model
    synthetic_aligned = synthetic_aligned[reduced.columns]
    # Save the aligned synthetic data
    output_aligned = 'D:/PythonProject/Result/Data_Preprocessing/synthetic_data_aligned.csv'
    synthetic_aligned.to_csv(output_aligned, index=False)
    print(f"Aligned synthetic data saved as: {output_aligned}")
    print(f"Aligned synthetic data shape: {synthetic_aligned.shape}")
    return synthetic_aligned

if __name__ == "__main__":
    processed_data = process_synthetic_data() 