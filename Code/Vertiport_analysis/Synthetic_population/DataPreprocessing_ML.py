import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load data
input_file = "/Result/check_ignor/Model_XgBoost/Synthetic_population/Mapping.csv"
data = pd.read_csv(input_file, low_memory=False)

print("=== SYNTHETIC POPULATION DATA PROCESSING SCRIPT ===")
print("=" * 60)
print(f"Input file: {input_file}")

# Print initial data info
print(f"Initial data shape: {data.shape}")
print(f"Initial columns: {data.columns.tolist()}")
print("\nData types:")
print(data.dtypes.value_counts())

# Check for missing values
print(f"\nMissing values per column:")
missing_counts = data.isnull().sum()
print(missing_counts[missing_counts > 0])

# Print sample data
print(f"\nFirst few rows:")
print(data.head())
print()

# Step 1: Define which columns to keep for processing
features_to_keep = [
    'person_id', 'age', 'gender', 'child_household', 'occupation',
    'adults_household', 'driversLicense', 'monthly_income',
    'autos', 'distance', 'autos_TT', 'PT_TT', 'travel_cost_auto', 'travel_cost_pt'
]

# Create a copy with only the features we want to process
data_processed = data[features_to_keep].copy()

print(f"Selected features: {len(features_to_keep)}")
print(f"Features: {features_to_keep}")
print()

# Step 2: Define column types more systematically
print("Defining column types...")

# Numerical: Continuous variables (travel times, costs, distance)
numerical_cols = [
    'distance', 'autos_TT', 'PT_TT', 'travel_cost_auto', 'travel_cost_pt'
]

# Ordinal: Categorical with inherent order (none in this case)
ordinal_cols = []

# Nominal: Categorical without inherent order - will be one-hot encoded
nominal_cols = [
    'age', 'gender', 'monthly_income', 'occupation', 'child_household', 
    'adults_household', 'autos', 'driversLicense'
]

# Reference: Keep for reference but not for ML
reference_cols = [
    'person_id'
]

print("Column types:")
print(f"  Numerical columns: {numerical_cols}")
print(f"  Ordinal columns: {ordinal_cols}")
print(f"  Nominal columns: {nominal_cols}")
print(f"  Reference columns: {reference_cols}")
print()

# Step 3: Handle missing values appropriately
print("Handling missing values...")

# For numerical: Fill with median
for col in numerical_cols:
    if data_processed[col].isnull().sum() > 0:
        median_val = data_processed[col].median()
        data_processed[col] = data_processed[col].fillna(median_val)
        print(f"  {col}: filled {data_processed[col].isnull().sum()} missing values with median {median_val:.2f}")

# For nominal: Fill with mode
for col in nominal_cols:
    if data_processed[col].isnull().sum() > 0:
        mode_val = data_processed[col].mode().iloc[0] if len(data_processed[col].mode()) > 0 else '0'
        data_processed[col] = data_processed[col].fillna(mode_val)
        print(f"  {col}: filled {data_processed[col].isnull().sum()} missing values with mode '{mode_val}'")

print("Missing values handled.")
print()

# Step 4: Separate features (X) - no target variable in this case
X = data_processed.drop(columns=reference_cols).copy()

print(f"Feature matrix shape: {X.shape}")
print()

# Step 5: Create preprocessing pipeline
print("Creating preprocessing pipeline...")

transformers = []

# Add nominal transformer if there are nominal columns
if nominal_cols:
    transformers.append(('nominal', Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), nominal_cols))

# Add numerical transformer - KEEP ORIGINAL VALUES (no scaling)
transformers.append(('numerical', Pipeline([
    ('passthrough', 'passthrough')  # Keep original values - NO STANDARDIZATION
]), numerical_cols))


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop'  # Drop any columns not explicitly handled
)

# Apply preprocessing
print("Applying preprocessing...")
X_processed = preprocessor.fit_transform(X)

# Get feature names
feature_names = []
if nominal_cols:
    nominal_names = preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)
    feature_names.extend(nominal_names)
feature_names.extend(numerical_cols)

print(f"Processed feature matrix shape: {X_processed.shape}")
print(f"Number of feature names: {len(feature_names)}")

# Create processed DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

print(f"Final processed data shape: {X_processed_df.shape}")
print(f"Feature names: {feature_names[:10]}...")  # Show first 10
print()


# Check for occupation columns
occupation_cols = [col for col in X_processed_df.columns if col.startswith('occupation_')]
print(f"Occupation columns found: {occupation_cols}")
print()

# Step 8: Rename columns 

# First, rename the occupation columns to employment
employment_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('occupation_'):
        # Map occupation values to employment categories
        if col == 'occupation_1':
            employment_rename_map[col] = 'employment_employed'
        elif col == 'occupation_2':
            employment_rename_map[col] = 'employment_unemployed'
        elif col == 'occupation_3':
            employment_rename_map[col] = 'employment_student'
        elif col == 'occupation_0':
            employment_rename_map[col] = 'employment_I prefer not to answer'
        else:
            # Handle any other occupation values
            employment_rename_map[col] = f'employment_other_{col.split("_")[1]}'

# Rename driversLicense columns to Driving_License
drivers_license_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('driversLicense_'):
        if col == 'driversLicense_0':
            drivers_license_rename_map[col] = 'driving_license_no'
        elif col == 'driversLicense_1':
            drivers_license_rename_map[col] = 'driving_license_yes'
        else:
            drivers_license_rename_map[col] = f'driving_license_other_{col.split("_")[1]}'

# Rename age columns
age_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('age_'):
        age_code = col.split("_")[1]
        if age_code == '0':
            age_rename_map[col] = 'age_missing'
        elif age_code == '1':
            age_rename_map[col] = 'age_1_17'
        elif age_code == '2':
            age_rename_map[col] = 'age_18_25'
        elif age_code == '3':
            age_rename_map[col] = 'age_26_35'
        elif age_code == '4':
            age_rename_map[col] = 'age_36_45'
        elif age_code == '5':
            age_rename_map[col] = 'age_46_55'
        elif age_code == '6':
            age_rename_map[col] = 'age_56_65'
        elif age_code == '7':
            age_rename_map[col] = 'age_65_plus'
        else:
            age_rename_map[col] = f'age_other_{age_code}'

# Rename gender columns
gender_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('gender_'):
        if col == 'gender_1':
            gender_rename_map[col] = 'female'
        elif col == 'gender_2':
            gender_rename_map[col] = 'male'
        elif col == 'gender_3':
            gender_rename_map[col] = 'diverse'
        else:
            gender_rename_map[col] = f'gender_other_{col.split("_")[1]}'

# Rename monthly_income columns
income_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('monthly_income_'):
        income_code = col.split("_")[2]
        if income_code == '0':
            income_rename_map[col] = 'income_no'  # prefer not to answer or no income
        elif income_code == '1':
            income_rename_map[col] = 'income_under1000'  # under €1000
        elif income_code == '2':
            income_rename_map[col] = 'income_1000-2000'  # €1000-2000
        elif income_code == '3':
            income_rename_map[col] = 'income_2000-3000'  # €2000-3000
        elif income_code == '4':
            income_rename_map[col] = 'income_3000-4000'  # €3000-4000
        elif income_code == '5':
            income_rename_map[col] = 'income_4000-5000'  # €4000-5000
        elif income_code == '6':
            income_rename_map[col] = 'income_5000-6000'  # €5000-6000
        elif income_code == '7':
            income_rename_map[col] = 'income_6000-7000'  # €6000-7000
        elif income_code == '8':
            income_rename_map[col] = 'income_7000_plus'  # €7000+
        else:
            income_rename_map[col] = f'income_other_{income_code}'

# Combine all rename mappings
all_rename_map = {**employment_rename_map, **drivers_license_rename_map, **age_rename_map, **gender_rename_map, **income_rename_map}

# Rename the columns
X_processed_df = X_processed_df.rename(columns=all_rename_map)

print("Column renaming completed:")
for old_name, new_name in all_rename_map.items():
    print(f"  '{old_name}' -> '{new_name}'")
print()

# Step 9: Add back the trip_id and spatial information for reference
# We'll add these as additional columns for potential use
reference_cols = ['trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY','departure_time','departure_time_return']
for col in reference_cols:
    if col in data.columns:
        X_processed_df[col] = data[col].values

# Step 7: Add trip_length in meters
if 'distance' in X_processed_df.columns:
    X_processed_df['trip_length'] = X_processed_df['distance'] * 1000

    # Reorder columns to place 'Trip_Length-m' after 'Trip_Length-km'
    cols = list(X_processed_df.columns)
    km_idx = cols.index('distance')

    # Remove 'Trip_Length-m' and insert after 'Trip_Length-km'
    cols.remove('trip_length')
    cols.insert(km_idx + 1, 'trip_length')
    cols.remove('distance')
    X_processed_df = X_processed_df[cols]

# Step 10: Save the processed data
output_file = "/Result/check_ignor/Model_XgBoost/Synthetic_population/DataPreprocessing_ML.csv"
X_processed_df.to_csv(output_file, index=False)

# Step 9: Print comprehensive summary
print("\n=== PROCESSING COMPLETED ===")
print(f"Output file: {output_file}")
print(f"Final shape: {X_processed_df.shape}")
print(f"Final columns: {list(X_processed_df.columns)}")
print()
print("Processing completed successfully!") 