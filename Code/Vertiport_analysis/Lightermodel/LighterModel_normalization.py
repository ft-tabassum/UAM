import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("=== LIGHTERMODEL NORMALIZATION SCRIPT ===")
print("=" * 50)

# Load the processed LighterModel data
input_file = "/Result/LighterModel/LighterModel_processing.csv"
data = pd.read_csv(input_file)

print(f"Input file: {input_file}")
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print()

# Replace any -1 values with 0 (as in the original UamNormalization)
data = data.replace(-1, 0)

# Drop all InVehicleTime and WaitingTime columns for all modes
cols_to_drop = [col for col in data.columns if 'InVehicleTime' in col or 'WaitingTime' in col]
data = data.drop(columns=cols_to_drop)

# Identify numerical features to normalize
# Exclude categorical columns and target variable
categorical_cols = []
numerical_cols = []

for col in data.columns:
    if col == 'tmode':  # Target variable - don't normalize
        continue
    elif col.startswith(
            ('Employment_', 'Driving_License_', 'disability_', 'Child_Household_', 'Adult_household_', 'autos_')):
        # One-hot encoded categorical columns
        categorical_cols.append(col)
    elif col in ['Gender', 'Age', 'Monthly_Income', 'purpose']:
        # Label encoded categorical columns
        categorical_cols.append(col)
    else:
        # Numerical columns (trip length, travel times, costs)
        numerical_cols.append(col)

print("Column classification:")
print(f"  Numerical columns to normalize: {numerical_cols}")
print(f"  Categorical columns (preserved): {categorical_cols}")
print(f"  Target variable: tmode")
print()

# Select features to normalize
features_to_normalize = data[numerical_cols]

print(f"Features to normalize shape: {features_to_normalize.shape}")
print(f"Features: {list(features_to_normalize.columns)}")
print()

# Initialize scaler and normalize
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features_to_normalize)

# Put normalized data back to DataFrame
data_normalized = pd.DataFrame(scaled_features, columns=features_to_normalize.columns)

# Add back categorical columns unchanged
for col in categorical_cols:
    data_normalized[col] = data[col].values

# Add back target variable unchanged
data_normalized['tmode'] = data['tmode']

print("Normalization completed!")
print(f"Normalized data shape: {data_normalized.shape}")
print()

# Display normalization statistics
print("=== NORMALIZATION STATISTICS ===")
for col in numerical_cols:
    original_mean = data[col].mean()
    original_std = data[col].std()
    normalized_mean = data_normalized[col].mean()
    normalized_std = data_normalized[col].std()

    print(f"{col}:")
    print(f"  Original - mean: {original_mean:.4f}, std: {original_std:.4f}")
    print(f"  Normalized - mean: {normalized_mean:.4f}, std: {normalized_std:.4f}")
print()

# Save normalized data
output_file = "/Result/LighterModel/LighterModel_normalized.csv"
data_normalized.to_csv(output_file, index=False)

print(f"Normalized data saved to: {output_file}")
print("Normalization completed successfully!") 