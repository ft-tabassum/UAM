import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

# Calculate rtts and rtts_weighted
data['rtts'] = 1 - (data['travel_time_Uam'] / data['autos_TT'])
data['rtts_weighted'] = data['rtts'] * data['prob_mode_Autonomous Flying Taxi']

# Filter for positive values
positive_rtts_data = data[data['rtts'] > 0]
positive_rtts_weighted_data = data[data['rtts_weighted'] > 0]

print("=" * 80)
print("ANALYSIS: Why Weighted RTTs Looks Different")
print("=" * 80)

print(f"Total trips: {len(data):,}")
print(f"Trips with positive rtts: {len(positive_rtts_data):,}")
print(f"Trips with positive rtts_weighted: {len(positive_rtts_weighted_data):,}")

print("\n" + "=" * 50)
print("RTTs Statistics:")
print("=" * 50)
print(f"RTTs range: {positive_rtts_data['rtts'].min():.4f} to {positive_rtts_data['rtts'].max():.4f}")
print(f"RTTs mean: {positive_rtts_data['rtts'].mean():.4f}")
print(f"RTTs median: {positive_rtts_data['rtts'].median():.4f}")

print("\n" + "=" * 50)
print("RTTs Weighted Statistics:")
print("=" * 50)
print(f"RTTs Weighted range: {positive_rtts_weighted_data['rtts_weighted'].min():.4f} to {positive_rtts_weighted_data['rtts_weighted'].max():.4f}")
print(f"RTTs Weighted mean: {positive_rtts_weighted_data['rtts_weighted'].mean():.4f}")
print(f"RTTs Weighted median: {positive_rtts_weighted_data['rtts_weighted'].median():.4f}")

print("\n" + "=" * 50)
print("Probability of UAM Statistics:")
print("=" * 50)
print(f"Probability range: {data['prob_mode_Autonomous Flying Taxi'].min():.4f} to {data['prob_mode_Autonomous Flying Taxi'].max():.4f}")
print(f"Probability mean: {data['prob_mode_Autonomous Flying Taxi'].mean():.4f}")
print(f"Probability median: {data['prob_mode_Autonomous Flying Taxi'].median():.4f}")

print("\n" + "=" * 50)
print("Key Insight:")
print("=" * 50)
print("RTTs Weighted = RTTs × Probability of UAM")
print("Since Probability of UAM is typically < 1.0,")
print("RTTs Weighted will always be smaller than RTTs")
print("This explains why the weighted values are lower!")

# Show some examples
print("\n" + "=" * 50)
print("Examples (first 10 positive rtts trips):")
print("=" * 50)
sample_data = positive_rtts_data.head(10)[['rtts', 'prob_mode_Autonomous Flying Taxi', 'rtts_weighted']]
print(sample_data.round(4))

print("\n" + "=" * 50)
print("Trip Length Analysis for Weighted RTTs:")
print("=" * 50)
# Filter for 20-180 km range
filtered_weighted = positive_rtts_weighted_data[
    (positive_rtts_weighted_data['trip_length'] >= 20000) & 
    (positive_rtts_weighted_data['trip_length'] <= 180000)
]

print(f"Filtered weighted trips (20-180 km): {len(filtered_weighted):,}")
if len(filtered_weighted) > 0:
    print(f"Weighted RTTs range in filtered data: {filtered_weighted['rtts_weighted'].min():.4f} to {filtered_weighted['rtts_weighted'].max():.4f}")
    print(f"Weighted RTTs mean in filtered data: {filtered_weighted['rtts_weighted'].mean():.4f}")

print("=" * 80)
