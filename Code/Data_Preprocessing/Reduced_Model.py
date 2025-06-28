import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

# Load data
data = pd.read_excel('D://Files_D//Study//Thesis//data//Discrete_data//uam_dataset.xlsx')

# Print original data info
print("Original data shape:", data.shape)
print("Original columns:", list(data.columns))
print("\nFirst few rows:")
print(data.head())

# Define the columns to keep with their new names (if renaming is needed)
# Based on the actual column names found in the dataset
columns_to_keep = {
    'Driving_License': 'driving license',
    'PT_Subscription': 'pt subscription', 
    'Reason': 'Purpose',  # renamed from 'Reason' to 'Purpose'
    'HouseholdCar': 'household car',
    'Gender': 'gender',
    'Age': 'age',
    'Child_Household': 'child_household',
    'Physical_Disabilities': 'disability',
    'Education': 'education',
    'Employment': 'employment',
    'Monthly_Income': 'monthly income',
    'tmode': 'mode',  # renamed from 'tmode' to 'mode'
    'tripLength': 'veh_time',  # assuming tripLength is the vehicle time
    'InVehicleTime_Car': 'InVehicleTime_Car',
    'WaitingTime_Car': 'WaitingTime_Car',
    'TravelCost_Car': 'TravelCost_Car',
    'Availability_Car': 'Availability_Car',
    'InVehicleTime_PublicTransport': 'InVehicleTime_PublicTransport',
    'WaitingTime_PublicTransport': 'WaitingTime_PublicTransport',
    'TravelCost_PublicTransport': 'TravelCost_PublicTransport',
    'Availability_PublicTransport': 'Availability_PublicTransport',
    'InVehicleTime_CarSharing': 'InVehicleTime_CarSharing',
    'WaitingTime_CarSharing': 'WaitingTime_CarSharing',
    'TravelCost_CarSharing': 'TravelCost_CarSharing',
    'Availability_CarSharing': 'Availability_CarSharing',
    'InVehicleTime_RideHailing': 'InVehicleTime_RideHailing',
    'WaitingTime_RideHailing': 'WaitingTime_RideHailing',
    'TravelCost_RideHailing': 'TravelCost_RideHailing',
    'Availability_RideHailing': 'Availability_RideHailing',
    'InVehicleTime_Uam': 'InVehicleTime_Uam',
    'WaitingTime_Uam': 'WaitingTime_Uam',
    'TravelCost_Uam': 'TravelCost_Uam'
}

# Check which columns exist in the original dataset
available_columns = []
missing_columns = []

for old_name, new_name in columns_to_keep.items():
    if old_name in data.columns:
        available_columns.append((old_name, new_name))
    else:
        missing_columns.append(old_name)

print(f"\nAvailable columns to keep: {len(available_columns)}")
print("Available columns:", [col[0] for col in available_columns])

print(f"\nMissing columns: {len(missing_columns)}")
if missing_columns:
    print("Missing columns:", missing_columns)

# Create reduced dataset with only the specified columns
reduced_data = data[[col[0] for col in available_columns]].copy()

# Rename columns as specified
column_mapping = {old: new for old, new in available_columns if old != new}
reduced_data = reduced_data.rename(columns=column_mapping)

print(f"\nReduced data shape: {reduced_data.shape}")
print("Reduced data columns:", list(reduced_data.columns))
print("\nFirst few rows of reduced data:")
print(reduced_data.head())

# Map tmode (now called 'mode') to numeric categories
tmode_mapping = {
    'Car': 0,
    'Public Transport': 1,
    'Car-sharing': 2,
    'Ride-hailing': 3,
    'UAM': 4
}
reduced_data['mode'] = reduced_data['mode'].map(tmode_mapping)
reduced_data['mode'] = reduced_data['mode'].fillna(0).astype(int)

# Print tmode mapping
print("\ntmode mapping:")
for k, v in tmode_mapping.items():
    print(f"  '{k}' -> {v}")
print()

# Define ordinal and nominal columns for the reduced dataset
# Based on the columns we have, we'll identify which are ordinal and which are nominal
ordinal_cols = []
nominal_cols = []

# Check each column type
for col in reduced_data.columns:
    if col == 'mode':  # Skip the target variable
        continue
    elif col in ['age', 'monthly income', 'veh_time']:  # Numeric columns that might be ordinal
        ordinal_cols.append(col)
    elif reduced_data[col].dtype == 'object':  # Categorical columns
        nominal_cols.append(col)
    else:  # Other numeric columns (like travel times, costs, availability)
        ordinal_cols.append(col)

print(f"Ordinal columns: {ordinal_cols}")
print(f"Nominal columns: {nominal_cols}")

# Replace missing nominal with '0' string
if nominal_cols:
    reduced_data[nominal_cols] = reduced_data[nominal_cols].fillna('0').astype(str)

# Replace missing ordinal with numeric 0
for col in ordinal_cols:
    reduced_data[col] = reduced_data[col].fillna(0)

# Label encode ordinal columns and print mappings
label_encoders = {}
for col in ordinal_cols:
    if reduced_data[col].dtype == 'object':  # Only encode if it's categorical
        le = LabelEncoder()
        reduced_data[col] = le.fit_transform(reduced_data[col])
        label_encoders[col] = le

        print(f"Label encoding mapping for '{col}':")
        for category, code in zip(le.classes_, range(len(le.classes_))):
            print(f"  '{category}' -> {code}")
        print()

# Separate features and target
X = reduced_data.drop(columns=['mode'])
y = reduced_data['mode']

# One-hot encode nominal variables
if nominal_cols:
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('nominal', categorical_transformer, nominal_cols)
    ], remainder='passthrough')

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
    X_processed_df = X.copy()
    X_processed_df['mode'] = y.values

print(f"Processed reduced data shape: {X_processed_df.shape}")
print("Processed reduced data columns:", list(X_processed_df.columns))
print("\nFirst few rows of processed reduced data:")
print(X_processed_df.head())

# Save the processed reduced dataset
output_path = '../../Result/Data_Preprocessing/reduced_data_processed.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
X_processed_df.to_csv(output_path, index=False)
print(f"\nProcessed reduced data saved to: {output_path}")

# Display basic statistics
print("\nBasic statistics of processed reduced dataset:")
print(X_processed_df.describe())

# Check for missing values
print("\nMissing values in processed reduced dataset:")
missing_values = X_processed_df.isnull().sum()
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
else:
    print("No missing values found")