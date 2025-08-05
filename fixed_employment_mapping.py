import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os


# Function to calculate Monthly_Income
def calculate_monthly_income(income):
    """ Monthly_Income = income * (1+0.0346)^13 * (1+0.2532)
    Where: - Inflation Rate = 25.32% - Annual Growth Rate = 3.46% - Years = 13 (from 2011 to 2024) """
    if pd.isna(income):
        return np.nan

    # Formula: income * (1 + annual_growth_rate)^years * (1 + inflation_rate)
    annual_growth_rate = 0.0346
    inflation_rate = 0.2532
    years = 13

    monthly_income = income * ((1 + annual_growth_rate) ** years) * (1 + inflation_rate)
    return monthly_income


# Function to categorize Monthly_Income
def categorize_monthly_income(monthly_income):
    """Categorize monthly income into predefined categories"""
    if pd.isna(monthly_income) or monthly_income <= 0:
        return 0  # 'I prefer not to answer'
    if monthly_income == 0:
        return 1  # 'No income'
    if monthly_income < 1000:
        return 2  # 'Under € 1000'
    elif monthly_income < 2000:
        return 3  # '€ 1000 to less than € 2000'
    elif monthly_income < 3000:
        return 4  # '€ 2000 to less than € 3000'
    elif monthly_income < 4000:
        return 5  # '€ 3000 to less than € 4000'
    elif monthly_income < 5000:
        return 6  # '€ 4000 to less than € 5000'
    elif monthly_income < 6000:
        return 7  # '€ 5000 to less than € 6000'
    elif monthly_income < 7000:
        return 8  # '€ 6000 to less than € 7000'
    else:
        return 9  # '€ 7000 or more'

# Function to apply mappings
def apply_mapping(df):
    """ Apply mapping to categorical variables according to the documentation"""
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

    # --- Occupation ---
    print("  - Mapping occupation...")
    print(f"    Original occupation unique values: {df['occupation'].unique() if 'occupation' in df.columns else 'Column not found'}")
    
    occupation_map = {
        'I prefer not to answer': 0,
        'Employed': 1,
        'Unemployed': 2,
        'Student': 3}
    df['occupation'] = df['occupation'].map(occupation_map).fillna(0).astype(int)
    
    print(f"    After mapping occupation unique values: {df['occupation'].unique()}")
    print(f"    Occupation value counts: {df['occupation'].value_counts().sort_index()}")

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

#---------------------------Data Preprocessing for ML Model---------------------------------------#
# Function to preprocess data for ML
def preprocess_for_ml(df):
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

    # Handle missing values
    data_processed[nominal_cols] = data_processed[nominal_cols].fillna('0').astype(str)
    for col in ordinal_cols:
        data_processed[col] = data_processed[col].fillna(0)
    for col in numerical_cols:
        data_processed[col] = data_processed[col].fillna(0)

    # **CRITICAL FIX**: Ensure all occupation categories are present
    print("Ensuring all occupation categories are present...")
    
    # Define all possible categories for each nominal column
    all_occupation_categories = ['0', '1', '2', '3']  # All possible occupation values as strings
    all_driversLicense_categories = ['0', '1']
    
    # For occupation - ensure all categories exist by adding dummy rows if needed
    current_occupation_categories = set(data_processed['occupation'].unique())
    missing_occupation_categories = set(all_occupation_categories) - current_occupation_categories
    
    if missing_occupation_categories:
        print(f"Missing occupation categories: {missing_occupation_categories}")
        print("Adding dummy rows to ensure all categories are represented...")
        
        # Create dummy rows for missing categories
        dummy_rows = []
        for missing_cat in missing_occupation_categories:
            dummy_row = data_processed.iloc[0].copy()  # Copy structure from first row
            dummy_row['occupation'] = missing_cat
            dummy_row['person_id'] = f'dummy_{missing_cat}'  # Mark as dummy
            dummy_rows.append(dummy_row)
        
        # Add dummy rows
        dummy_df = pd.DataFrame(dummy_rows)
        data_processed = pd.concat([data_processed, dummy_df], ignore_index=True)
        print(f"Added {len(dummy_rows)} dummy rows")

    # Label Encoding for ordinal columns
    label_encoders = {}
    for col in ordinal_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
        label_encoders[col] = le

    # One-Hot Encoding for nominal columns with all categories specified
    if nominal_cols:
        print(f"Current occupation categories before encoding: {sorted(data_processed['occupation'].unique())}")
        
        # Create OneHotEncoder with specified categories to ensure consistency
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(
                categories=[
                    all_occupation_categories,  # occupation
                    all_driversLicense_categories,  # driversLicense  
                    'auto',  # autos - let it auto-detect
                    'auto',  # child_household - let it auto-detect
                    'auto'   # adults_household - let it auto-detect
                ],
                handle_unknown='ignore', 
                sparse_output=False
            ))
        ])
        
        preprocessor = ColumnTransformer(transformers=[('nominal', categorical_transformer, nominal_cols)],
                                         remainder='passthrough')
        X_processed = preprocessor.fit_transform(data_processed)

        # Get one-hot encoded feature names
        onehot_feature_names = list(
            preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)
        )

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
    
    # Remove dummy rows if they were added
    if missing_occupation_categories:
        print("Removing dummy rows...")
        dummy_mask = X_processed_df['person_id'].astype(str).str.startswith('dummy_')
        X_processed_df = X_processed_df[~dummy_mask].reset_index(drop=True)
        print(f"Removed {dummy_mask.sum()} dummy rows")
    
    # Verify all occupation columns exist
    occupation_cols = [col for col in X_processed_df.columns if col.startswith('occupation_')]
    print(f"Final occupation columns: {occupation_cols}")
    for col in occupation_cols:
        print(f"  {col}: {X_processed_df[col].sum()} occurrences")
    
    return X_processed_df

# Function for renaming columns according to LighterModel naming convention
def rename_columns(df):
    print("Renaming columns to match LighterModel naming convention...")
    occupation_cols = [col for col in df.columns if col.startswith('occupation_')]
    print(f"Found occupation columns: {occupation_cols}")

    # First, rename the occupation columns to Employment
    employment_rename_map = {}
    for col in df.columns:
        if col.startswith('occupation_'):  # Map occupation values to Employment categories
            if col == 'occupation_0':
                employment_rename_map[col] = 'Employment_I prefer not to answer'
            elif col == 'occupation_1':
                employment_rename_map[col] = 'Employment_Employed'
            elif col == 'occupation_2':
                employment_rename_map[col] = 'Employment_Unemployed'
            elif col == 'occupation_3':
                employment_rename_map[col] = 'Employment_Student'

    # Rename driversLicense columns to Driving_License
    drivers_license_rename_map = {}
    for col in df.columns:
        if col.startswith('driversLicense_'):
            if col == 'driversLicense_0':
                drivers_license_rename_map[col] = 'Driving_License_No'
            elif col == 'driversLicense_1':
                drivers_license_rename_map[col] = 'Driving_License_Yes'
    # Combine all rename mappings
    all_rename_map = {**employment_rename_map, **drivers_license_rename_map}

    # Rename the columns
    df = df.rename(columns=all_rename_map)

    print("Column renaming completed:")
    for old_name, new_name in all_rename_map.items():
            print(f"  '{old_name}' -> '{new_name}'")

    return df

# Add reference columns and trip length calculations
def add_reference_columns_and_trip_length(df, X_processed_df):

    print("Adding reference columns and trip length...")

    # Add additional columns if they exist in original dataframe
    reference_cols = ['trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY']
    for col in reference_cols:
        if col in df.columns:
            X_processed_df[col] = df[col].values

    # After renaming columns and before saving the processed data
    if 'distance' in X_processed_df.columns:
        X_processed_df['tripLength'] = X_processed_df['distance'] * 1000 #unit: m

        # Reorder columns to place 'tripLength-m' after 'distance'
        cols = list(X_processed_df.columns)
        distance_idx = cols.index('distance')

        # Remove 'tripLength' and insert after 'distance'
        cols.remove('tripLength')
        cols.insert(distance_idx + 1, 'tripLength')
        X_processed_df = X_processed_df[cols]

    return X_processed_df


# Main function to read, process, and save data
def main():
    print("=== COMBINED INCOME CALCULATION AND MAPPING SCRIPT ===")
    print("=" * 60)

    input_file = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/microdata_trips.csv"
    output_file = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/merger.csv"

    try:
        # Read the input data
        df = pd.read_csv(input_file)
        print(f"Input data shape: {df.shape}")

        # Step 1: Apply income calculation and mappings
        df['Annual_Income'] = df['income'].apply(calculate_monthly_income)
        df['Monthly_Income_value'] = df['Annual_Income'] / 12.0
        df['Monthly_Income'] = df['Monthly_Income_value'].apply(categorize_monthly_income)
        df = apply_mapping(df)

        # Step 2: Preprocess data for ML
        df_processed = preprocess_for_ml(df)

        # Step 3: Rename columns
        X_processed_df = rename_columns(df_processed)

        # Step 4: Add reference columns and trip length
        df_processed = add_reference_columns_and_trip_length(df, X_processed_df)

        # Step 5: Final verification
        employment_cols = [col for col in df_processed.columns if col.startswith('Employment_')]
        print(f"\nFinal Employment columns: {employment_cols}")
        for col in employment_cols:
            print(f"  {col}: {df_processed[col].sum()} occurrences")

        # Step 6: Save processed data
        df_processed.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        print(f"final data shape: {df_processed.shape}")


    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()