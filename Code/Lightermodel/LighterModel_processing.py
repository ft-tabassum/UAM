import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load data
data = pd.read_excel("D://Files_D//Study//Thesis//data//Lighter_UAM//Lighter_UAM.xlsx")

# Step 1: Drop Unnamed columns (remove columns like 'Unnamed: 33')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Step 2: Drop Education column explicitly
data = data.drop(columns=['Education'], errors='ignore')

# Step 3: tmode mapping
tmode_mapping = {
    'Car': 0,
    'Public Transport': 1,
    'Car-sharing': 2,
    'Ride-hailing': 3,
    'UAM': 4
}
data['tmode'] = data['tmode'].map(tmode_mapping)
data['tmode'] = data['tmode'].fillna(0).astype(int)

# Print tmode mapping
print("tmode mapping:")
for k, v in tmode_mapping.items():
    print(f"  '{k}' -> {v}")
print()

# Step 4: Map Employment to 4 categories before processing
employment_mapping = {
    'Working full-time': 'Employed',
    'Working part-time': 'Employed',
    'Employment_Other': 'Employed',
    'Unemployed': 'Unemployed',
    'Retired': 'Unemployed',
    'Housewife or househusband': 'Unemployed',
    'Pupil, student or apprentice/intern': 'Student',
    'I prefer not to answer': 'I prefer not to answer'
}

# Apply employment mapping
data['Employment'] = data['Employment'].map(employment_mapping)
data['Employment'] = data['Employment'].fillna('I prefer not to answer')

print("Employment mapping applied:")
for k, v in employment_mapping.items():
    print(f"  '{k}' -> '{v}'")
print()

# Step 5: Define ordinal and nominal columns
ordinal_cols = ['Monthly_Income', 'Gender', 'Age', 'purpose']
nominal_cols = data.select_dtypes(include=['object']).columns.tolist()
nominal_cols = [col for col in nominal_cols if col not in ordinal_cols and col != 'tmode' and col != 'Employment']  # Exclude Employment from nominal_cols

# Step 6: Replace missing values
# Replace missing nominal with '0' string
data[nominal_cols] = data[nominal_cols].fillna('0').astype(str)

# Replace missing ordinal with numeric 0
for col in ordinal_cols:
    data[col] = data[col].fillna(0)

# Step 7: Custom mapping for Gender and Age to match the synthetic population script
print("Custom mapping for Gender and Age to match synthetic population script:")
print()

# Custom Gender mapping
gender_mapping = {
    'Female': 1,
    'Male': 2,
    'Diverse': 3
}
data['Gender'] = data['Gender'].map(gender_mapping).fillna(3).astype(int)
print("Gender mapping:")
for k, v in gender_mapping.items():
    print(f"  '{k}' -> {v}")
print()

# Custom Age mapping
age_mapping = {
    'missing': 0,
    '1-17': 1,
    '18-29': 2,
    '30-39': 3,
    '40-49': 4,
    '50-59': 5,
    '60-69': 6,
    '70-79': 7,
    'I prefer not to answer': 8
}
# Apply age mapping
data['Age'] = data['Age'].map(age_mapping).fillna(0).astype(int)
print("Age mapping:")
for k, v in age_mapping.items():
    print(f"  '{k}' -> {v}")
print()

# Label Encoding for remaining ordinal columns (Monthly_Income and purpose)
remaining_ordinal_cols = [col for col in ordinal_cols if col not in ['Gender', 'Age']]
label_encoders = {}
for col in remaining_ordinal_cols:
    # Convert to string to ensure consistent data types
    data[col] = data[col].astype(str)

    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

    print(f"Label encoding mapping for '{col}':")
    for category, code in zip(le.classes_, range(len(le.classes_))):
        print(f"  '{category}' -> {code}")
    print()

# Step 8: Separate Features (X) and Target (y)
X = data.drop(columns=['tmode'])
y = data['tmode']

# Step 9: One-Hot Encoding of Nominal Columns and Employment
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('employment', categorical_transformer, ['Employment']),
    ('nominal', categorical_transformer, nominal_cols)
], remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# Step 10: Get the feature names after One-Hot Encoding
passthrough_cols = [col for col in X.columns if col not in nominal_cols and col != 'Employment']
feature_names = list(preprocessor.named_transformers_['employment'].named_steps['onehot'].get_feature_names_out(['Employment'])) + \
                list(preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)) + passthrough_cols

# Convert to dense array if sparse
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()

# Step 11: Convert the processed data to a DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Step 12: Add the target column (tmode) to the final DataFrame
X_processed_df['tmode'] = y.values

# Step 13: Save the processed data to a CSV file
output_file = "/Code/Lightermodel/Result_LM/LighterModel_processing.csv"
X_processed_df.to_csv(output_file, index=False)
print(f"Processed data saved to '{output_file}'")