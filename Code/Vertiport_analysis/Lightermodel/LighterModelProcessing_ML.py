import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

# Load data of Survey
# Features that are similar to "Synthetic_population" are considered here
data = pd.read_excel("D://Files_D//Study//Thesis//data//Lighter_UAM//Lighter_UAM.xlsx")

print("Original data shape:", data.shape)
print("Original columns:", data.columns.tolist())

# 1) Drop Education column
if 'Education' in data.columns:
    data = data.drop('Education', axis=1)
    print("Dropped Education column")

# 2) Convert 'tripLength' to meters and create 'trip_length'
if 'tripLength' in data.columns:
    # Assuming tripLength is in kilometers, convert to meters
    data['trip_length'] = data['tripLength'] * 1000
    data.drop('tripLength', axis=1, inplace=True)
    print("Created trip_length column (converted from km to meters)")

# 3) 'tmode' mapping
tmode_mapping = {
    'Car': 0,
    'Public Transport': 1,
    'Car-sharing': 2,
    'Ride-hailing': 3,
    'UAM': 4
}
if 'tmode' in data.columns:
    data['tmode'] = data['tmode'].map(tmode_mapping)
    print("Mapped tmode values")

# 4) 'Employment' mapping
employment_mapping = {
    'Working full-time': 'Employed',
    'Working part-time': 'Employed',
    'Unemployed': 'Unemployed',
    'Retired': 'Unemployed',
    'Housewife or househusband': 'Unemployed',
    'Pupil, student or apprentice/intern': 'Student',
    'I prefer not to answer': 'I prefer not to answer'
}
if 'Employment' in data.columns:
    data['Employment'] = data['Employment'].map(employment_mapping)
    print("Mapped Employment values")

# 5) 'purpose' mapping
purpose_mapping = {
    'Business trip': 'Business_trip',
    'Medical travel': 'Medical_trip',
    'Other': 'Other_trip',
    'Tourism': 'Tourism_trip',
    'Visiting family or friends': 'Visit_trip'
}
if 'purpose' in data.columns:
    data['purpose'] = data['purpose'].map(purpose_mapping)
    print("Mapped purpose values")

# 6) 'autos' mapping
autos_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3 or more': 3
}
if 'autos' in data.columns:
    data['autos'] = data['autos'].map(autos_mapping).fillna(0)
    print("Mapped autos values")

# 7) 'Child_Household' mapping
child_household_mapping = {
    "I don't have children": 0,
    'I prefer not to answer': 0,
    '1': 1,
    '2': 2,
    '3': 3
}
if 'Child_Household' in data.columns:
    data['Child_Household'] = data['Child_Household'].map(child_household_mapping)
    print("Mapped Child_Household values")

# 8) 'disability' mapping
disability_mapping = {
    'Yes': 1,
    'No': 0,
    'I prefer not to answer': 0
}
if 'disability' in data.columns:
    data['disability'] = data['disability'].map(disability_mapping)
    print("Mapped disability values")

# 9) 'Adult_household' mapping
adult_household_mapping = {
    '1': 1,
    '2': 2,
    '3': 3,
    '4 or more': 4,
    'I prefer not to answer': 0
}
if 'Adult_household' in data.columns:
    data['Adult_household'] = data['Adult_household'].map(adult_household_mapping)
    print("Mapped Adult_household values")

# 10) 'Driving_License' mapping
driving_license_mapping = {
    'Yes': 1,
    'No': 0
}
if 'Driving_License' in data.columns:
    data['Driving_License'] = data['Driving_License'].map(driving_license_mapping)
    print("Mapped Driving_License values")

# 11) 'Gender' mapping
gender_mapping = {
    'Female': 1,
    'Male': 2,
    'Diverse': 3
}
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map(gender_mapping).astype(int)
    print("Mapped Gender values")

# 12) 'Age' mapping
age_mapping = {
    "17": 1,
    "18-29": 2,
    "30-39": 3,
    "40-49": 4,
    "50-59": 5,
    "60-69": 6,
    "70-79": 7,
    'I prefer not to answer': 8
}
if 'Age' in data.columns:
    # First check if there are any unmapped values
    unmapped_values = data['Age'].unique()
    print(f"Unique Age values in data: {unmapped_values}")

    # Map the values and fill any unmapped with default
    data['Age'] = data['Age'].map(age_mapping).fillna(8).astype(int)
    print("Mapped Age values")

# 13) 'Monthly_Income' mapping
monthly_income_mapping = {
    '€ 1000 to less than € 2000': 3,
    '€ 2000 to less than € 3000': 4,
    '€ 3000 to less than € 4000': 5,
    '€ 4000 to less than € 5000': 6,
    '€ 5000 to less than € 6000': 7,
    '€ 6000 to less than € 7000': 8,
    '€ 7000 or more': 9,
    'I prefer not to answer': 0,
    'No income': 1,
    'Under € 1000': 2
}
if 'Monthly_Income' in data.columns:
    data['Monthly_Income'] = data['Monthly_Income'].map(monthly_income_mapping).astype(int)
    print("Mapped Monthly_Income values")

# 14) Define column types
ordinal_columns = ['Monthly_Income', 'Gender', 'Age']
nominal_columns = ['tmode', 'Employment', 'disability', 'Driving_License', 'autos', 'Child_Household',
                   'Adult_household', 'purpose']
numerical_columns = ['trip_length', 'travel_time_auto', 'in_vehicle_time_auto', 'waiting_time_auto', 'travel_cost_auto',
                     'travel_time_pt', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_cost_pt',
                     'travel_time_CarSharing', 'in_vehicle_time_CarSharing', 'waiting_time_CarSharing',
                     'travel_cost_CarSharing',
                     'travel_time_RideHailing', 'in_vehicle_time_RideHailing', 'waiting_time_RideHailing',
                     'travel_cost_RideHailing',
                     'travel_time_Uam', 'in_vehicle_time_Uam', 'waiting_time_Uam', 'travel_cost_Uam']

# 15) Rename columns
column_rename_mapping = {
    'disability': 'Disability',
    'Child_Household': 'child_household',
    'Adult_household': 'adults_household',
    'Monthly_Income': 'monthly_income',
    'Age': 'age',
    'Gender': 'gender'
}

data = data.rename(columns=column_rename_mapping)
print("Renamed columns")

# Update column lists after renaming
ordinal_columns = ['monthly_income', 'gender', 'age']
nominal_columns = ['tmode', 'Employment', 'Disability', 'Driving_License', 'autos', 'child_household',
                   'adults_household', 'purpose']

# 16) Replace missing values with 0
data = data.fillna(0)
print("Replaced missing values with 0")

# 17) Separate Features (X) and Target (y)
y = data['tmode']  # Target variable
X = data.drop('tmode', axis=1)  # Features

print(f"Target shape: {y.shape}")
print(f"Features shape: {X.shape}")

# 18) Apply one-hot encoding to nominal columns
# First, let's check which nominal columns exist in the data
available_nominal_columns = [col for col in nominal_columns if col in X.columns]
print(f"Available nominal columns for encoding: {available_nominal_columns}")

# Create a copy of X for encoding
X_encoded = X.copy()

# Apply one-hot encoding to nominal columns
if available_nominal_columns:
    # Convert all nominal columns to strings to ensure consistent data types
    for col in available_nominal_columns:
        if col == 'autos':
            # Convert autos to integers first, then to strings to avoid float suffixes
            X_encoded[col] = X_encoded[col].astype(int).astype(str)
        else:
            X_encoded[col] = X_encoded[col].astype(str)

    # Create one-hot encoder
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform nominal columns
    nominal_encoded = onehot_encoder.fit_transform(X_encoded[available_nominal_columns])

    # Get feature names from the encoder
    feature_names = onehot_encoder.get_feature_names_out(available_nominal_columns)

    # Create DataFrame with encoded features
    nominal_df = pd.DataFrame(nominal_encoded, columns=feature_names, index=X_encoded.index)

    # Print the actual column names to see what OneHotEncoder created
    print(f"OneHotEncoder created columns: {nominal_df.columns.tolist()}")

    # Rename one-hot encoded columns to make them more meaningful
    column_rename_map = {}
    for col in nominal_df.columns:
        # Disability column renaming
        if col == 'Disability_0':
            column_rename_map[col] = 'Disability_No'
        elif col == 'Disability_1':
            column_rename_map[col] = 'Disability_Yes'
        # Driving_License column renaming
        elif col == 'Driving_License_0':
            column_rename_map[col] = 'Driving_License_No'
        elif col == 'Driving_License_1':
            column_rename_map[col] = 'Driving_License_Yes'
        # All other columns remain unchanged (no else clause)

    # Apply the renaming
    nominal_df = nominal_df.rename(columns=column_rename_map)

    # Remove duplicate Employment_0 column if it exists (since we have Employment_I prefer not to answer)
    if 'Employment_0' in nominal_df.columns:
        nominal_df = nominal_df.drop('Employment_0', axis=1)
        print("Removed duplicate Employment_0 column")

    # Drop original nominal columns and concatenate with encoded ones
    X_encoded = X_encoded.drop(available_nominal_columns, axis=1)
    X_encoded = pd.concat([X_encoded, nominal_df], axis=1)

    print(f"Applied one-hot encoding to {len(available_nominal_columns)} nominal columns")
    print(f"Encoded features shape: {X_encoded.shape}")

# Final data preparation
print("\n=== Final Data Summary ===")
print(f"Final X shape: {X_encoded.shape}")
print(f"Final y shape: {y.shape}")
print(f"Number of features: {X_encoded.shape[1]}")
print(f"Target classes: {y.unique()}")

# Save data as single CSV
# Add target variable back to the processed data
LighterModelProcessing_ML = X_encoded.copy()
LighterModelProcessing_ML['tmode'] = y

LighterModelProcessing_ML.to_csv("D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/LighterModel/LighterModelProcessing_ML.csv", index=False)

print("\nLighterModelProcessing_ML saved to:")
print("- LighterModelProcessing_ML.csv")

# Display sample of processed data
print("\n=== Sample of Processed Features ===")
print(X_encoded.head())
print("\n=== Sample of Target ===")
print(y.head())