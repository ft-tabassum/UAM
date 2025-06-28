import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load data
data = pd.read_excel('D://Files_D//Study//Thesis//data//Discrete_data//uam_dataset.xlsx')

# Drop unnecessary columns
cols_to_drop = [
    "question", "id", "uniqueId", "version","chosen",
    "TB03A[other]", "TB03B[other]", "TB11[other]", "Country",
    "SD15[other]", "PostalCode", "interviewtime",
    "RP", "SP", "givenUamCost", "av_Car", "av_PublicTransport",
    "av_CarSharing", "av_RideHailing", "av_Uam", "Trip_Start"
]
data = data.drop(columns=cols_to_drop, errors='ignore')

# Map tmode to numeric categories
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

# Define ordinal and nominal columns
ordinal_cols = ['Age', 'Monthly_Income', 'tripLength', 'Gender']
nominal_cols = data.select_dtypes(include=['object']).columns.tolist()
nominal_cols = [col for col in nominal_cols if col not in ordinal_cols and col != 'tmode']

# Replace missing nominal with '0' string
data[nominal_cols] = data[nominal_cols].fillna('0').astype(str)

# Replace missing ordinal with numeric 0
for col in ordinal_cols:
    data[col] = data[col].fillna(0)

# Label encode ordinal columns and print mappings
label_encoders = {}
for col in ordinal_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

    print(f"Label encoding mapping for '{col}':")
    for category, code in zip(le.classes_, range(len(le.classes_))):
        print(f"  '{category}' -> {code}")
    print()

# Separate features and target
X = data.drop(columns=['tmode'])
y = data['tmode']

# One-hot encode nominal variables
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
X_processed_df['tmode'] = y.values

print(f"Processed data shape: {X_processed_df.shape}")
print(X_processed_df.head())

X_processed_df.to_csv('Uamdata_processed.csv', index=False)
print("Processed data saved to 'Uamdata_processed.csv'")
