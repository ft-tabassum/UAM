import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("=== TRIAL SYNTHETIC NORMALIZATION SCRIPT ===")
print("=" * 55)

# Load the processed trial synthetic data
input_file = "../../../LargeFiles_synthetic/synthetic_population_processing.csv"
data = pd.read_csv(input_file)

print(f"Input file: {input_file}")
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print()

# Replace any -1 values with 0 (as in the original UamNormalization)
data = data.replace(-1, 0)

# Identify numerical features to normalize
# Exclude categorical columns and spatial reference columns
categorical_cols = []
numerical_cols = []
spatial_cols = []

for col in data.columns:
    if col in ['trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY']:
        # Spatial reference columns - preserve as is
        spatial_cols.append(col)
    elif col.startswith(('Employment_', 'Driving_License_', 'disability_', 'Child_Household_', 'Adult_household_', 'autos_')):
        # One-hot encoded categorical columns
        categorical_cols.append(col)
    elif col in ['person_id', 'age', 'gender', 'Monthly_Income', 'purpose']:
        # Label encoded categorical columns and person_id
        categorical_cols.append(col)
    else:
        # Numerical columns (trip length, travel times, costs)
        numerical_cols.append(col)

print("Column classification:")
print(f"  Numerical columns to normalize: {numerical_cols}")
print(f"  Categorical columns (preserved): {len(categorical_cols)} columns")
print(f"  Spatial reference columns (preserved): {spatial_cols}")
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

# Add back spatial reference columns unchanged
for col in spatial_cols:
    data_normalized[col] = data[col].values

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
output_file = "../../../LargeFiles_synthetic/trial_normalized.csv"
data_normalized.to_csv(output_file, index=False)

print(f"Normalized data saved to: {output_file}")
print("Normalization completed successfully!") 