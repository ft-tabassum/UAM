import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

# Load data
input_file = "../../../LargeFiles_synthetic/trial_micro_trips_with_income_mapping.csv"
data = pd.read_csv(input_file)

print("=== TRIAL DATA PROCESSING SCRIPT ===")
print("=" * 60)
print(f"Input file: {input_file}")
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print()

# Step 1: Define which columns to keep for processing
# We'll keep the key features and drop some spatial/trip-specific columns
features_to_keep = [
    'person_id', 'age', 'gender', 'Child_Household', 'occupation', 
    'Adult_household', 'driversLicense', 'Monthly_Income', 'disability', 'purpose', 
    'autos', 'distance', 'time_auto', 'time_PT', 
    'TravelCost_auto', 'TravelCost_PT'
]

# Create a copy with only the features we want to process
data_processed = data[features_to_keep].copy()

print(f"Selected features: {len(features_to_keep)}")
print(f"Features: {features_to_keep}")
print()

# Step 2: Define ordinal and nominal columns
# Ordinal columns (categorical with inherent order)
ordinal_cols = ['Monthly_Income', 'age', 'gender', 'purpose']

# Nominal columns (categorical without inherent order) - will be one-hot encoded
nominal_cols = ['occupation', 'driversLicense', 'autos', 'Child_Household', 'Adult_household', 'disability']

# Numerical columns (continuous variables)
numerical_cols = [
    'distance', 'time_auto', 'time_PT', 'TravelCost_auto', 'TravelCost_PT'
]

print("Column types:")
print(f"  Ordinal columns: {ordinal_cols}")
print(f"  Nominal columns: {nominal_cols}")
print(f"  Numerical columns: {numerical_cols}")
print()

# Step 3: Replace missing values
print("Handling missing values...")

# Replace missing nominal with '0' string
data_processed[nominal_cols] = data_processed[nominal_cols].fillna('0').astype(str)

# Replace missing ordinal with numeric 0
for col in ordinal_cols:
    data_processed[col] = data_processed[col].fillna(0)

# Replace missing numerical with 0
for col in numerical_cols:
    data_processed[col] = data_processed[col].fillna(0)

print("Missing values handled.")
print()

# Step 4: Custom mapping for Gender to match LighterModel
print("Applying custom mapping for Gender...")
gender_mapping = {
    1: 1,  # Female
    2: 2,  # Male
    3: 3   # Diverse
}
data_processed['gender'] = data_processed['gender'].map(gender_mapping).fillna(3).astype(int)
print("Gender mapping:")
for k, v in gender_mapping.items():
    print(f"  {k} -> {v}")
print()

# Step 5: Label Encoding of remaining Ordinal Columns
print("Applying Label Encoding to remaining ordinal columns...")
remaining_ordinal_cols = [col for col in ordinal_cols if col != 'gender']
label_encoders = {}

for col in remaining_ordinal_cols:
    # Convert to string to ensure consistent data types
    data_processed[col] = data_processed[col].astype(str)

    le = LabelEncoder()
    data_processed[col] = le.fit_transform(data_processed[col])
    label_encoders[col] = le

    print(f"Label encoding mapping for '{col}':")
    for category, code in zip(le.classes_, range(len(le.classes_))):
        print(f"  '{category}' -> {code}")
    print()

# Step 6: Separate Features (X) - no target variable in this case
X = data_processed.copy()

print(f"Feature matrix shape: {X.shape}")
print()

# Step 7: One-Hot Encoding of Nominal Columns
print("Applying One-Hot Encoding to nominal columns...")

if nominal_cols:
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('nominal', categorical_transformer, nominal_cols)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)

    # Get the feature names after One-Hot Encoding
    passthrough_cols = [col for col in X.columns if col not in nominal_cols]
    feature_names = list(preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)) + passthrough_cols

    # Convert to dense array if sparse
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    # Convert the processed data to a DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
else:
    # If no nominal columns, just use the original data
    X_processed_df = X.copy()
    feature_names = list(X.columns)

print(f"Processed feature matrix shape: {X_processed_df.shape}")
print(f"Feature names: {feature_names}")
print()



# Step 8: Rename columns to match LighterModel naming convention
print("Renaming columns to match LighterModel...")

# First, rename the occupation columns to Employment
employment_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('occupation_'):
        # Map occupation values to Employment categories
        if col == 'occupation_1':
            employment_rename_map[col] = 'Employment_Employed'
        elif col == 'occupation_2':
            employment_rename_map[col] = 'Employment_Unemployed'
        elif col == 'occupation_3':
            employment_rename_map[col] = 'Employment_Student'
        else:
            employment_rename_map[col] = 'Employment_I prefer not to answer'

# Rename driversLicense columns to Driving_License
drivers_license_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('driversLicense_'):
        if col == 'driversLicense_0':
            drivers_license_rename_map[col] = 'Driving_License_No'
        elif col == 'driversLicense_1':
            drivers_license_rename_map[col] = 'Driving_License_Yes'
        else:
            drivers_license_rename_map[col] = 'Driving_License_Unknown'

# Rename disability columns
disability_rename_map = {}
for col in X_processed_df.columns:
    if col.startswith('disability_'):
        if col == 'disability_0':
            disability_rename_map[col] = 'disability_No'
        else:
            disability_rename_map[col] = 'disability_Yes'

# Create a mapping for other column renaming
column_rename_map = {
    'distance': 'tripLength-km',
    'time_auto': 'travel time_car',
    'time_PT': 'travel time_PublicTransport',
    'TravelCost_auto': 'TravelCost_Car',
    'TravelCost_PT': 'TravelCost_PublicTransport'
}

# Combine all rename mappings
all_rename_map = {**employment_rename_map, **drivers_license_rename_map, **disability_rename_map, **column_rename_map}

# Rename the columns
X_processed_df = X_processed_df.rename(columns=all_rename_map)

print("Column renaming completed:")
for old_name, new_name in all_rename_map.items():
    print(f"  '{old_name}' -> '{new_name}'")
print()

# Step 9: Add back the trip_id and spatial information for reference
# We'll add these as additional columns for potential use
reference_cols = ['trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY']
for col in reference_cols:
    if col in data.columns:
        X_processed_df[col] = data[col].values

# Step 10: Save the processed data
output_file = "../../../LargeFiles_synthetic/trial_processed.csv"
X_processed_df.to_csv(output_file, index=False)

print("=== PROCESSING COMPLETED ===")
print(f"Output file: {output_file}")
print(f"Final shape: {X_processed_df.shape}")
print(f"Final columns: {list(X_processed_df.columns)}")
print()

# Step 11: Display summary statistics
print("=== DATA SUMMARY ===")
print(f"Total records: {len(X_processed_df)}")
print(f"Total features: {len(X_processed_df.columns)}")

# Show some statistics for key numerical features
numerical_features = ['tripLength-km', 'travel time_car', 'travel time_PublicTransport', 'TravelCost_Car', 'TravelCost_PublicTransport']
for feature in numerical_features:
    if feature in X_processed_df.columns:
        print(f"  {feature}: mean={X_processed_df[feature].mean():.2f}, std={X_processed_df[feature].std():.2f}")

print()
print("Processing completed successfully!") 