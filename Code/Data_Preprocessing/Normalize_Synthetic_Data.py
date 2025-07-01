import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def normalize_synthetic_data():
    """
    Normalize aligned synthetic data for probability calculations.
    The 'mode' column (target) is not normalized. Missing values and -1 are replaced with 0.
    """
    print("Normalizing aligned synthetic data...")
    # Load data
    data = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/synthetic_data_aligned.csv')
    # Replace NaN and -1 with 0
    data = data.fillna(0)
    data = data.replace(-1, 0)
    # Exclude the target column from normalization
    features = data.drop(columns=['mode'])
    # Only keep columns that are present in the aligned synthetic data
    # (all except 'mode')
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    data_normalized = pd.DataFrame(scaled_features, columns=features.columns)
    data_normalized['mode'] = data['mode']
    # Save normalized data
    output_path = 'D:/PythonProject/Result/Data_Preprocessing/synthetic_data_normalized.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_normalized.to_csv(output_path, index=False)
    print(f"Normalized data saved: {output_path}")
    print("Aligned synthetic data normalization completed successfully!")
    print(f"Normalized data shape: {data_normalized.shape}")
    print("\nAligned synthetic data statistics after normalization:")
    print(f"Shape: {data_normalized.shape}")
    mode_counts = data_normalized['mode'].value_counts().sort_index()
    print("Mode distribution:")
    for mode, count in mode_counts.items():
        print(f"  Mode {mode}: {count} ({count/len(data_normalized)*100:.1f}%)")
    print(f"\nFeature ranges after MinMax normalization:")
    for col in features.columns[:5]:  # Show first 5 features
        min_val = data_normalized[col].min()
        max_val = data_normalized[col].max()
        print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
    missing_values = data_normalized.isnull().sum().sum()
    print(f"\nMissing values: {missing_values}")
    return data_normalized, scaler

if __name__ == "__main__":
    normalized_data, scaler = normalize_synthetic_data() 