import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/Result/DataPreprocessing_aft/aft_processed.csv')

print("Original data shape:", data.shape)
print("Original data info:")
print(data.info())

# Define feature categories
numerical_cols = [
    'CAR_CO', 'CAR_INC', 'CAR_TT', 'PT_CO', 'PT_INC', 'PT_TT',
    'AFT_CO', 'AFT_INC', 'AFT_TT'
]

ordinal_cols = [
    'AtoLattitude_r1', 'AtoLattitude_r2', 'AtoLattitude_r3', 'AtoLattitude_r4',
    'Likelihood_r1', 'Likelihood_r2', 'Likelihood_r3', 'Likelihood_r4', 'Likelihood_r5', 'Likelihood_r6',
    'technologyconcern_r1', 'technologyconcern_r2', 'technologyconcern_r3', 'technologyconcern_r4',
    'environmentconcern_r1', 'environmentconcern_r2', 'environmentconcern_r3', 'environmentconcern_r4',
    'satisfaction'
]

safety_multimodal_cols = [
    'CAR_SAFETY_ds', 'PT_MULTI_inpart', 'PT_SAFETY_safer', 
    'AFT_MULTI_yes', 'AFT_SAFETY_ds', 'AFT_SAFETY_riskier', 'AFT_SAFETY_safer'
]

# Get all other columns (demographics, one-hot encoded)
other_cols = [col for col in data.columns if col not in numerical_cols + ordinal_cols + safety_multimodal_cols + ['CHOICE']]

print(f"\nFeature categorization:")
print(f"Numerical features: {len(numerical_cols)} - {numerical_cols}")
print(f"Ordinal features: {len(ordinal_cols)} - {ordinal_cols}")
print(f"Safety/Multi-modal features: {len(safety_multimodal_cols)} - {safety_multimodal_cols}")
print(f"Other features: {len(other_cols)}")

# Show original statistics
print(f"\nOriginal data statistics:")
print("Numerical features:")
print(data[numerical_cols].describe())

print(f"\nOrdinal features (should be 0-4 scale):")
print(data[ordinal_cols].describe())

print(f"\nSafety/Multi-modal features (should be binary 0/1):")
print(data[safety_multimodal_cols].describe())

# OPTION 1: Selective normalization (recommended)
print(f"\n=== OPTION 1: Selective Normalization ===")
print("Only normalize numerical features (costs, times), keep others as-is")

data_selective = data.copy()

# Only normalize numerical features
if numerical_cols:
    scaler = MinMaxScaler()
    data_selective[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    print(f"Normalized {len(numerical_cols)} numerical features using MinMaxScaler")

# Show results
print(f"\nAfter selective normalization:")
print("Numerical features (now 0-1 scale):")
print(data_selective[numerical_cols].describe())

print(f"\nOrdinal features (still 0-4 scale):")
print(data_selective[ordinal_cols].describe())

print(f"\nSafety/Multi-modal features (still binary 0/1):")
print(data_selective[safety_multimodal_cols].describe())

# Save selective normalization
data_selective.to_csv("D:/Thesis/UAM/Result/DataPreprocessing_aft/aft_normalized.csv", index=False)
print(f"\nSelective normalization saved to: aft_normalized.csv")




