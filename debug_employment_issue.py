import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def debug_employment_mapping(df):
    """Debug function to check employment mapping issues"""
    print("=== DEBUGGING EMPLOYMENT MAPPING ===")
    print(f"Original occupation column unique values:")
    if 'occupation' in df.columns:
        print(df['occupation'].value_counts(dropna=False))
        print(f"Data types: {df['occupation'].dtype}")
    else:
        print("No 'occupation' column found!")
    
    print("\n" + "="*50)
    return df

def apply_mapping_debug(df):
    """Modified apply_mapping function with debug output"""
    print("Applying mappings...")

    # ------- Age--------
    print("  - Mapping age...")
    def bin_age(age):
        if pd.isna(age):
            return 0  # 'missing'
        try:
            age = int(age)
        except:
            return 0  # 'missing'
        if 1 <= age <= 17:
            return 1  # '1-17'
        elif 18 <= age <= 29:
            return 2  # '18-29'
        elif 30 <= age <= 39:
            return 3  # '30-39'
        elif 40 <= age <= 49:
            return 4  # '40-49'
        elif 50 <= age <= 59:
            return 5  # '50-59'
        elif 60 <= age <= 69:
            return 6  # '60-69'
        elif 70 <= age <= 79:
            return 7  # '70-79'
        else:
            return 8  # 'I prefer not to answer'

    df['age'] = df['age'].apply(bin_age)

    # --- Gender-----
    print("  - Mapping gender...")
    gender_map = {'Male': 2, 'Female': 1, 'Diverse': 3}
    df['gender'] = df['gender'].map(gender_map).fillna(3).astype(int)

    # --- Occupation with DEBUG ---
    print("  - Mapping occupation...")
    print(f"    Original occupation values: {df['occupation'].unique()}")
    print(f"    Original occupation counts:")
    print(df['occupation'].value_counts(dropna=False))
    
    occupation_map = {
        'I prefer not to answer': 0,
        'Employed': 1,
        'Unemployed': 2,
        'Student': 3}
    
    # Check which values won't be mapped
    unmapped_values = set(df['occupation'].unique()) - set(occupation_map.keys()) - {np.nan}
    if unmapped_values:
        print(f"    WARNING: Unmapped occupation values: {unmapped_values}")
    
    df['occupation'] = df['occupation'].map(occupation_map).fillna(0).astype(int)
    
    print(f"    After mapping occupation values: {df['occupation'].unique()}")
    print(f"    After mapping occupation counts:")
    print(df['occupation'].value_counts(dropna=False))

    # --- driversLicense  ---
    print("  - Mapping driversLicense...")
    df['driversLicense'] = df['driversLicense'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(
        0).astype(int)

    # --- Disability ---
    print("  - Mapping disability...")
    df['disability'] = df['disability'].map({0: 0, 1: 1}).fillna(0).astype(int)
    
    # --- Purpose ---
    print("  - Mapping purpose with new categories...")
    purpose_conversion = {
        1: 'Business trip',  # HBW -> Business trip
        2: 'Business trip',  # HBE -> Business trip
        3: 'Visiting family or friends',  # HBS -> Visiting family or friends
        4: 'Tourism',  # HBR -> Tourism
        5: 'Other',  # HBO -> Other
        6: 'Business trip',  # NHBW -> Business trip
        7: 'Other'  # NHBO -> Other
    }
    df['purpose_category'] = df['purpose'].map(purpose_conversion).fillna('Other')

    # Now map categories to final codes
    purpose_final_map = {
        'Business trip': 0,
        'Medical travel': 1,
        'Other': 2,
        'Tourism': 3,
        'Visiting family or friends': 4
    }
    df['purpose'] = df['purpose_category'].map(purpose_final_map).fillna(2).astype(int)
    df = df.drop(columns=['purpose_category'])

    return df

def preprocess_for_ml_debug(df):
    print("=== DATA PROCESSING SCRIPT ===")
    print("=" * 60)
    print("Available columns in DataFrame:")
    print(df.columns.tolist())
    print()

    # Define columns to keep for processing
    features_to_keep = [
        'person_id', 'age', 'gender', 'child_household', 'occupation', 'adults_household', 'driversLicense',
        'Monthly_Income', 'disability', 'purpose', 'autos', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
        'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto',
        'travel_cost_pt'
    ]

    # Create a copy with only the features we want to process
    data_processed = df[features_to_keep].copy()

    # Define ordinal and nominal columns
    ordinal_cols = ['Monthly_Income', 'age', 'gender', 'purpose']
    nominal_cols = ['occupation', 'driversLicense', 'autos', 'child_household', 'adults_household']
    numerical_cols = ['disability', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
        'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto',
                      'travel_cost_pt']

    # DEBUG: Check occupation values before preprocessing
    print("BEFORE PREPROCESSING:")
    print(f"Occupation unique values: {data_processed['occupation'].unique()}")
    print(f"Occupation value counts:")
    print(data_processed['occupation'].value_counts(dropna=False))

    # Handle missing values
    data_processed[nominal_cols] = data_processed[nominal_cols].fillna('0').astype(str)
    for col in ordinal_cols:
        data_processed[col] = data_processed[col].fillna(0)
    for col in numerical_cols:
        data_processed[col] = data_processed[col].fillna(0)

    # DEBUG: Check occupation values after missing value handling
    print("\nAFTER MISSING VALUE HANDLING:")
    print(f"Occupation unique values: {data_processed['occupation'].unique()}")
    print(f"Occupation value counts:")
    print(data_processed['occupation'].value_counts(dropna=False))

    # Label Encoding for ordinal columns
    label_encoders = {}
    for col in ordinal_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
        label_encoders[col] = le

    # One-Hot Encoding for nominal columns
    if nominal_cols:
        print(f"\nApplying One-Hot Encoding to: {nominal_cols}")
        
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[('nominal', categorical_transformer, nominal_cols)],
                                         remainder='passthrough')
        X_processed = preprocessor.fit_transform(data_processed)

        # Get one-hot encoded feature names
        onehot_feature_names = list(
            preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)
        )

        # DEBUG: Print one-hot encoded feature names
        print(f"\nOne-hot encoded feature names: {onehot_feature_names}")
        occupation_features = [name for name in onehot_feature_names if name.startswith('occupation_')]
        print(f"Occupation features specifically: {occupation_features}")

        # Get the remaining column names (passthrough columns)
        passthrough_cols = [col for col in data_processed.columns if col not in nominal_cols]

        # Combine all feature names
        feature_names = onehot_feature_names + passthrough_cols

    else:
        X_processed = data_processed.copy()
        feature_names = list(data_processed.columns)

    # Convert to dense array if sparse
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    # Convert to DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # DEBUG: Check final occupation columns
    print(f"\nFINAL COLUMNS: {X_processed_df.columns.tolist()}")
    occupation_cols = [col for col in X_processed_df.columns if col.startswith('occupation_')]
    print(f"Final occupation columns: {occupation_cols}")
    
    for col in occupation_cols:
        print(f"{col} sum: {X_processed_df[col].sum()}")
    
    return X_processed_df

# Test function to run on a sample
def test_with_sample_data():
    """Create sample data to test the mapping"""
    sample_data = pd.DataFrame({
        'person_id': [1, 2, 3, 4, 5],
        'age': [25, 35, 45, 55, 65],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Diverse'],
        'occupation': ['Employed', 'Student', 'Unemployed', 'I prefer not to answer', 'Employed'],
        'child_household': [0, 1, 2, 0, 1],
        'adults_household': [2, 2, 1, 2, 3],
        'driversLicense': [True, False, True, True, False],
        'Monthly_Income': [5, 3, 1, 0, 4],
        'disability': [0, 0, 1, 0, 0],
        'purpose': [1, 3, 2, 4, 1],
        'autos': [1, 0, 1, 2, 0],
        'distance': [10.5, 5.2, 15.0, 8.3, 12.1],
        'in_vehicle_time_auto': [20, 15, 30, 18, 25],
        'waiting_time_auto': [5, 3, 8, 4, 6],
        'travel_time_auto': [25, 18, 38, 22, 31],
        'in_vehicle_time_pt': [35, 25, 45, 28, 40],
        'waiting_time_pt': [10, 8, 12, 9, 11],
        'travel_time_pt': [45, 33, 57, 37, 51],
        'travel_cost_auto': [5.50, 3.20, 8.00, 4.80, 6.10],
        'travel_cost_pt': [2.50, 2.50, 3.00, 2.50, 2.50]
    })
    
    print("=== TESTING WITH SAMPLE DATA ===")
    print("Original sample data:")
    print(sample_data[['person_id', 'occupation']].head())
    
    # Apply the debug mapping
    mapped_data = apply_mapping_debug(sample_data)
    print(f"\nAfter mapping, occupation values: {mapped_data['occupation'].unique()}")
    
    # Apply preprocessing
    processed_data = preprocess_for_ml_debug(mapped_data)
    
    return processed_data

if __name__ == "__main__":
    # Run test with sample data
    result = test_with_sample_data()
    print("\n" + "="*60)
    print("FINAL RESULT COLUMNS:")
    print([col for col in result.columns if 'occupation' in col.lower() or 'employment' in col.lower()])