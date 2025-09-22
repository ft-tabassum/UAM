import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv(
    '/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

# Calculate rtts for UAM
# rtts = 1 - (UAM_TT / ground-based, motorized mode (i.e., car or PT) TT)
data['rtts'] = 1 - (data['travel_time_Uam'] / data['autos_TT'])  # autos
data['rtts_weighted'] = data['rtts'] * data['prob_mode_Autonomous Flying Taxi']

# Filter trips where rtts is positive
positive_rtts_data = data[data['rtts'] > 0]
positive_rtts_weighted_data = data[data['rtts_weighted'] > 0]

# Count trips with positive rtts and rtts > 0.5
total_trips = len(data)
positive_rtts_count = len(positive_rtts_data)
high_rtts_count = len(data[data['rtts'] > 0.5])
high_rtts_weighted_count = len(data[data['rtts_weighted'] > 0.5])
total_trips_weighted = len(data)
positive_rtts_weighted_count = len(positive_rtts_weighted_data)
average_rtts_weighted_all = data['rtts_weighted'].mean()
average_rtts_weighted_positive = positive_rtts_weighted_data['rtts_weighted'].mean()

# Calculate average rtts for all trips
average_rtts_all = data['rtts'].mean()
average_rtts_positive = positive_rtts_data['rtts'].mean()
average_rtts_weighted_positive = positive_rtts_weighted_data['rtts_weighted'].mean()
average_rtts_weighted_all = data['rtts_weighted'].mean()

print("=" * 60)
print("RTTs STATISTICS SUMMARY")
print("=" * 60)
print(f"Total trips in dataset: {total_trips:,}")
print(f"Trips with positive rtts: {positive_rtts_count:,} ({positive_rtts_count/total_trips*100:.1f}%)")
print(f"Trips with rtts > 0.5: {high_rtts_count:,} ({high_rtts_count/total_trips*100:.1f}%)")
print(f"Average rtts for ALL trips: {average_rtts_all:.4f}")
print(f"Average rtts for positive rtts trips: {average_rtts_positive:.4f}")
print("=" * 60)

# First, let's examine the data range
print("Data range analysis:")
print(f"Trip length range: {positive_rtts_data['trip_length'].min()/1000:.1f} - {positive_rtts_data['trip_length'].max()/1000:.1f} km")
print(f"rtts range: {positive_rtts_data['rtts'].min():.3f} - {positive_rtts_data['rtts'].max():.3f}")

# Filter data to match reference image range (20-170 km)
filtered_data = positive_rtts_data[
    (positive_rtts_data['trip_length'] >= 20000) &  # 20 km in meters
    (positive_rtts_data['trip_length'] <= 170000)   # 170 km in meters
]

print(f"Filtered data points (20-170 km): {len(filtered_data)}")

# Create smaller bins for smoother curve (10 km bins from 20-170 km)
trip_length_bins = pd.cut(filtered_data['trip_length'], 
                         bins=range(20000, 171000, 10000),  # 10km bins from 20-170km
                         labels=[f"{i}-{i+10}" for i in range(20, 170, 10)])

binned_data = filtered_data.groupby(trip_length_bins).agg({
    'rtts': ['mean', 'std', 'count'],
    'trip_length': 'mean'
}).reset_index()

# Flatten column names
binned_data.columns = ['trip_length_bin', 'rtts_mean', 'rtts_std', 'count', 'trip_length_mean']

# Filter out bins with very few data points (less than 5 for statistical reliability)
binned_data = binned_data[binned_data['count'] >= 5]

print(f"Bins with sufficient data: {len(binned_data)}")
print("Binned data preview:")
print(binned_data[['trip_length_bin', 'trip_length_mean', 'rtts_mean', 'rtts_std', 'count']].round(3))

# ===== FIRST OUTPUT: RTTs vs Trip Length =====
plt.figure(figsize=(14, 10))

# Plot only the markers without the line and without error bars - dark green points (no border)
plt.plot(binned_data['trip_length_mean'] / 1000, binned_data['rtts_mean'], 
         color='darkgreen', linewidth=3.5, marker='o', markersize=8, 
          markerfacecolor='darkgreen', markeredgecolor='none',
         markeredgewidth=0, zorder=2, label='Mean rtts')

# Set title to match reference format
plt.title('Travel Time Savings for Trips (rtts)', fontsize=16, fontweight='bold', pad=20)

# Set axis labels
plt.xlabel('Trip Length (km)', fontsize=14, fontweight='bold')
plt.ylabel('Travel Time Savings Ratio (rtts)', fontsize=14, fontweight='bold')

# Set axis limits with more space around the data
plt.xlim(15, 175)  # Extended range with more space
plt.ylim(-0.02, 0.42)  # Extended range with more space

# Set x-axis intervals from 20 to 170 with 20 km increments
plt.xticks(range(20, 171, 20), fontsize=12)  # 20, 40, 60, 80, 100, 120, 140, 160

# Set y-axis intervals from 0.00 to 0.40 with 0.05 increments (keep same ruler)
plt.yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], fontsize=12)

# Add grid with light gray color to match reference
plt.grid(True, alpha=0.3, color='lightgray', linestyle='-', linewidth=0.5)

# Add legend on the left side
plt.legend(fontsize=12, loc='upper left')

# Adjust layout and save
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/rtts_vs_trip_length.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()  # Close the figure to free memory

# ===== SECOND OUTPUT: RTTs Weighted vs Trip Length =====
# Filter data for weighted rtts (20-170 km)
filtered_data_weighted = positive_rtts_weighted_data[
    (positive_rtts_weighted_data['trip_length'] >= 20000) &  # 20 km in meters
    (positive_rtts_weighted_data['trip_length'] <= 170000)   # 170 km in meters
]

print(f"Filtered weighted data points (20-170 km): {len(filtered_data_weighted)}")

# Create bins for weighted rtts
trip_length_bins_weighted = pd.cut(filtered_data_weighted['trip_length'], 
                         bins=range(20000, 171000, 10000),  # 10km bins from 20-170km
                         labels=[f"{i}-{i+10}" for i in range(20, 170, 10)])

binned_data_weighted = filtered_data_weighted.groupby(trip_length_bins_weighted).agg({
    'rtts_weighted': ['mean', 'std', 'count'],
    'trip_length': 'mean'
}).reset_index()

# Flatten column names
binned_data_weighted.columns = ['trip_length_bin', 'rtts_weighted_mean', 'rtts_weighted_std', 'count', 'trip_length_mean']

# Filter out bins with very few data points
binned_data_weighted = binned_data_weighted[binned_data_weighted['count'] >= 5]

print(f"Weighted bins with sufficient data: {len(binned_data_weighted)}")
print("Weighted binned data preview:")
print(binned_data_weighted[['trip_length_bin', 'trip_length_mean', 'rtts_weighted_mean', 'rtts_weighted_std', 'count']].round(3))

# Create second plot for weighted rtts
plt.figure(figsize=(14, 10))

# Plot weighted rtts
plt.plot(binned_data_weighted['trip_length_mean'] / 1000, binned_data_weighted['rtts_weighted_mean'], 
         color='darkblue', linewidth=3.5, marker='o', markersize=8, 
          markerfacecolor='darkblue', markeredgecolor='none',
         markeredgewidth=0, zorder=2, label='Mean rtts_weighted')

# Set title for weighted plot
plt.title('Weighted Travel Time Savings for Trips (rtts_weighted)', fontsize=16, fontweight='bold', pad=20)

# Set axis labels
plt.xlabel('Trip Length (km)', fontsize=14, fontweight='bold')
plt.ylabel('Weighted Travel Time Savings Ratio (rtts_weighted)', fontsize=14, fontweight='bold')

# Set axis limits with more space around the data
plt.xlim(15, 175)  # Extended range with more space
# Use appropriate y-axis scaling for weighted values (much smaller)
max_weighted = binned_data_weighted['rtts_weighted_mean'].max()
plt.ylim(-0.002, max_weighted * 1.2)  # Scale based on actual weighted data

# Set x-axis intervals from 20 to 170 with 20 km increments
plt.xticks(range(20, 171, 20), fontsize=12)  # 20, 40, 60, 80, 100, 120, 140, 160

# Set y-axis intervals based on weighted data range
y_max = max_weighted * 1.2
y_ticks = np.linspace(0, y_max, 9)  # Create 9 evenly spaced ticks
plt.yticks(y_ticks, [f"{tick:.3f}" for tick in y_ticks], fontsize=12)

# Add grid with light gray color to match reference
plt.grid(True, alpha=0.3, color='lightgray', linestyle='-', linewidth=0.5)

# Add legend on the left side
plt.legend(fontsize=12, loc='upper left')

# Adjust layout and save
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/rtts_weighted_vs_trip_length.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()  # Close the figure to free memory

# Print summary statistics
print("=" * 50)
print("SCRIPT EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 50)
print("Summary of binned data (rtts):")
print(binned_data[['trip_length_mean', 'rtts_mean', 'rtts_std', 'count']].round(3))
print(f"\nTotal number of rtts data points used: {len(positive_rtts_data)}")
print(f"Number of rtts bins with sufficient data: {len(binned_data)}")

print("\nSummary of binned data (rtts_weighted):")
print(binned_data_weighted[['trip_length_mean', 'rtts_weighted_mean', 'rtts_weighted_std', 'count']].round(3))
print(f"\nTotal number of rtts_weighted data points used: {len(positive_rtts_weighted_data)}")
print(f"Number of rtts_weighted bins with sufficient data: {len(binned_data_weighted)}")

print("=" * 50)
print("TWO GRAPHS GENERATED:")
print("1. rtts_vs_trip_length.png (dark green points)")
print("2. rtts_weighted_vs_trip_length.png (dark blue points)")
print("=" * 50)
