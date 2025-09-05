import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load data
data = pd.read_excel('D:/Files_D/Study/Thesis/new_data/aft_2ndversion.xlsx')

# Print initial data info
print("Initial data shape:", data.shape)
print("Initial columns:", data.columns.tolist())
print("\nData types:")
print(data.dtypes.value_counts())

# Check original CHOICE values
print(f"\nOriginal CHOICE values:")
print(data['CHOICE'].value_counts().sort_index())
print(f"CHOICE data type: {data['CHOICE'].dtype}")
print(f"Sample CHOICE values: {data['CHOICE'].head(10).tolist()}")

# Drop unnecessary columns
cols_to_drop = ["sys_RespNum", "missingvalue"]
data = data.drop(cols_to_drop, axis=1, errors='ignore')

# Map choice - handle different possible formats
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

# Check for missing values
print(f"\nMissing values per column:")
missing_counts = data.isnull().sum()
print(missing_counts[missing_counts > 0])

# Define column types more carefully
# Numerical: Costs, travel times, etc.
numerical_cols = [
    'CAR_CO', 'CAR_INC', 'CAR_TT', 'PT_CO', 'PT_INC', 'PT_TT',
    'AFT_CO', 'AFT_INC', 'AFT_TT'
]

# Ordinal: Likert scales (1-5 or similar)
ordinal_cols = [
    'AtoLattitude_r1', 'AtoLattitude_r2', 'AtoLattitude_r3', 'AtoLattitude_r4',
    'Likelihood_r1', 'Likelihood_r2', 'Likelihood_r3', 'Likelihood_r4', 'Likelihood_r5', 'Likelihood_r6',
    'technologyconcern_r1', 'technologyconcern_r2', 'technologyconcern_r3', 'technologyconcern_r4',
    'environmentconcern_r1', 'environmentconcern_r2', 'environmentconcern_r3', 'environmentconcern_r4',
    'satisfaction'
]

# Important safety and multi-modal preference features (binary 0/1)
safety_multimodal_cols = [
    'CAR_SAFETY_ds', 'PT_MULTI_inpart', 'PT_SAFETY_safer', 
    'AFT_MULTI_yes', 'AFT_SAFETY_ds', 'AFT_SAFETY_riskier', 'AFT_SAFETY_safer'
]

# Nominal: Any object/string columns that are not ordinal
nominal_cols = data.select_dtypes(include=['object']).columns.tolist()
nominal_cols = [col for col in nominal_cols if col not in ordinal_cols and col != 'CHOICE']

# One-hot: Already encoded demographics (binary 0/1, keep as-is)
one_hot_cols = [col for col in data.columns 
                if col not in ordinal_cols + numerical_cols + nominal_cols + safety_multimodal_cols + ['CHOICE']]

# Handle missing values appropriately
print(f"\nHandling missing values...")

# For numerical: Fill with median
for col in numerical_cols:
    if data[col].isnull().sum() > 0:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
        print(f"  {col}: filled {data[col].isnull().sum()} missing values with median {median_val:.2f}")

# For ordinal: Fill with median (treat as numeric)
for col in ordinal_cols:
    if data[col].isnull().sum() > 0:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
        print(f"  {col}: filled {data[col].isnull().sum()} missing values with median {median_val:.2f}")

# For safety/multi-modal: Fill with 0 (assuming missing means absence of that preference)
for col in safety_multimodal_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(0)
        print(f"  {col}: filled {data[col].isnull().sum()} missing values with 0 (no preference)")

# For nominal: Fill with mode
for col in nominal_cols:
    if data[col].isnull().sum() > 0:
        mode_val = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'Unknown'
        data[col] = data[col].fillna(mode_val)
        print(f"  {col}: filled {data[col].isnull().sum()} missing values with mode '{mode_val}'")

# For one-hot: Fill with 0 (assuming missing means absence)
for col in one_hot_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(0)
        print(f"  {col}: filled {data[col].isnull().sum()} missing values with 0")

# Convert ordinal columns to numeric (they should remain as numeric, not label encoded)
for col in ordinal_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    # Fill any remaining NaN with median
    if data[col].isnull().sum() > 0:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
    
    # Shift from 1-5 scale to 0-4 scale for Likert responses
    if col in ['AtoLattitude_r1', 'AtoLattitude_r2', 'AtoLattitude_r3', 'AtoLattitude_r4',
               'Likelihood_r1', 'Likelihood_r2', 'Likelihood_r3', 'Likelihood_r4', 'Likelihood_r5', 'Likelihood_r6',
               'technologyconcern_r1', 'technologyconcern_r2', 'technologyconcern_r3', 'technologyconcern_r4',
               'environmentconcern_r1', 'environmentconcern_r2', 'environmentconcern_r3', 'environmentconcern_r4',
               'satisfaction']:
        # Shift from 1-5 to 0-4 scale
        data[col] = data[col] - 1
        print(f"  Shifted {col} from 1-5 scale to 0-4 scale")

# Separate features and target
X = data.drop(columns=['CHOICE'])
y = data['CHOICE']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create preprocessing pipeline - OPTION 1: Keep original values (no standardization)
print(f"\nCreating preprocessing pipeline (keeping original values)...")

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

# Add ordinal transformer (keep as numeric)
transformers.append(('ordinal', Pipeline([
    ('passthrough', 'passthrough')
]), ordinal_cols))

# Add safety/multi-modal transformer (keep as binary 0/1)
transformers.append(('safety_multimodal', Pipeline([
    ('passthrough', 'passthrough')
]), safety_multimodal_cols))

# Add one-hot transformer (keep as-is)
transformers.append(('one_hot', Pipeline([
    ('passthrough', 'passthrough')
]), one_hot_cols))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop'  # Drop any columns not explicitly handled
)

# Apply preprocessing
print(f"\nApplying preprocessing...")
X_processed = preprocessor.fit_transform(X)

# Get feature names
feature_names = []
if nominal_cols:
    nominal_names = preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)
    feature_names.extend(nominal_names)
feature_names.extend(numerical_cols)
feature_names.extend(ordinal_cols)
feature_names.extend(safety_multimodal_cols)
feature_names.extend(one_hot_cols)

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

# Check for any remaining issues
print(f"\nFinal data info:")
print(f"Missing values: {X_processed_df.isnull().sum().sum()}")
print(f"Data types: {X_processed_df.dtypes.value_counts()}")

# Verify no unwanted columns
unwanted_cols = ['sys_RespNum', 'missingvalue']
found_unwanted = [col for col in X_processed_df.columns if col in unwanted_cols]
if found_unwanted:
    print(f"WARNING: Found unwanted columns: {found_unwanted}")
else:
    print("✓ No unwanted columns found")

# Save to CSV
output_path = '/Result/DataPreprocessing_aft/aft_processed.csv'
X_processed_df.to_csv(output_path, index=False)
print(f"\nProcessed data saved to '{output_path}'")

# Print some statistics
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

print(f"\nNumerical features statistics (original values):")
print(X_processed_df[numerical_cols].describe())

print(f"\nSample of processed data:")
print(X_processed_df[['CAR_CO', 'CAR_TT', 'PT_CO', 'PT_TT', 'AFT_CO', 'AFT_TT', 'CHOICE']].head())

# Show important safety and multi-modal features
print(f"\nSafety and Multi-modal preference features:")
for col in safety_multimodal_cols:
    if col in X_processed_df.columns:
        print(f"  {col}: {X_processed_df[col].value_counts().to_dict()}")

print(f"\nSample of safety/multi-modal features:")
safety_sample_cols = [col for col in safety_multimodal_cols if col in X_processed_df.columns]
if safety_sample_cols:
    print(X_processed_df[safety_sample_cols + ['CHOICE']].head())