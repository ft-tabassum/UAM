import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Load data
data = pd.read_csv('data_processed.csv')

# Replace -1 with 0
data = data.replace(-1, 0)

# Select features to normalize (exclude categorical 'tmode')
features = data.drop(columns=['tmode'])

# Initialize scaler and normalize
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Put normalized data back to DataFrame
data_normalized = pd.DataFrame(scaled_features, columns=features.columns)

# Add back 'tmode' column unchanged
data_normalized['tmode'] = data['tmode']

# Save or use data_normalized for  ML or analysis
data_normalized.to_csv('data_normalized.csv', index=False)

