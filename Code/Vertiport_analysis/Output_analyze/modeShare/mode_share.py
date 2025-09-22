import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv(
    '/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

print("=" * 60)
print("MODE SHARE ANALYSIS")
print("=" * 60)

# Step 1: Determine the chosen mode for each trip based on probabilities
def determine_mode(row):
    if row['prob_mode_Car'] > row['prob_mode_Public Transport'] and row['prob_mode_Car'] > row['prob_mode_Autonomous Flying Taxi']:
        return 'Car'
    elif row['prob_mode_Public Transport'] > row['prob_mode_Car'] and row['prob_mode_Public Transport'] > row['prob_mode_Autonomous Flying Taxi']:
        return 'Public Transport'
    elif row['prob_mode_Autonomous Flying Taxi'] > row['prob_mode_Car'] and row['prob_mode_Autonomous Flying Taxi'] > row['prob_mode_Public Transport']:
        return 'UAM'
    else:
        return 'Tie'  # Handle cases where probabilities are equal

# Apply the function to determine mode for each trip
data['chosen_mode'] = data.apply(determine_mode, axis=1)

# Step 2: Count the number of trips by each mode
mode_counts = data['chosen_mode'].value_counts()

# Step 3: Calculate the mode share as percentage of total trips
total_trips = len(data)
mode_shares = mode_counts / total_trips * 100

# Print out the mode share percentages
print("Mode Share Percentages:")
print(mode_shares)
print("\nMode Share Counts:")
print(mode_counts)

print("\n" + "=" * 60)
print("DETAILED ANALYSIS")
print("=" * 60)

# Additional analysis: Check probability ranges
print("\nProbability Statistics:")
print(f"Car probability range: {data['prob_mode_Car'].min():.4f} - {data['prob_mode_Car'].max():.4f}")
print(f"Public Transport probability range: {data['prob_mode_Public Transport'].min():.4f} - {data['prob_mode_Public Transport'].max():.4f}")
print(f"UAM probability range: {data['prob_mode_Autonomous Flying Taxi'].min():.4f} - {data['prob_mode_Autonomous Flying Taxi'].max():.4f}")

print(f"\nAverage probabilities:")
print(f"Car: {data['prob_mode_Car'].mean():.4f}")
print(f"Public Transport: {data['prob_mode_Public Transport'].mean():.4f}")
print(f"UAM: {data['prob_mode_Autonomous Flying Taxi'].mean():.4f}")

# Check for ties
ties = data[data['chosen_mode'] == 'Tie']
print(f"\nNumber of ties (equal probabilities): {len(ties)}")

# Create a bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(mode_counts.index, mode_counts.values, 
               color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(mode_counts)])
plt.title('Mode Share - Number of Trips', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Transportation Mode', fontsize=14, fontweight='bold')
plt.ylabel('Number of Trips', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(height):,}', ha='center', va='bottom', fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/mode_share_bar_chart.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Create a separate chart for percentage visualization
plt.figure(figsize=(12, 8))
bars = plt.bar(mode_shares.index, mode_shares.values, 
               color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(mode_shares)])
plt.title('Mode Share - Percentage Distribution', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Transportation Mode', fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/mode_share_percentage_chart.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Save results to CSV
mode_share_results = pd.DataFrame({
    'Mode': mode_counts.index,
    'Count': mode_counts.values,
    'Percentage': mode_shares.values
})
mode_share_results.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/mode_share_results.csv', 
                         index=False)

print("\n" + "=" * 60)
print("MODE SHARE BY TRIP DISTANCE")
print("=" * 60)

# Define distance categories
def categorize_distance(trip_length):
    if trip_length < 20000:  # < 20 km
        return 'Short (< 20km)'
    elif trip_length < 50000:  # 20-50 km
        return 'Medium (20-50km)'
    elif trip_length < 100000:  # 50-100 km
        return 'Long (50-100km)'
    else:  # > 100 km
        return 'Very Long (>100km)'

# Add distance category to data
data['distance_category'] = data['trip_length'].apply(categorize_distance)

# Analyze mode share by distance category
print("\nMode Share by Distance Category:")
print("-" * 50)

distance_categories = ['Short (< 20km)', 'Medium (20-50km)', 'Long (50-100km)', 'Very Long (>100km)']
mode_share_by_distance = {}

for category in distance_categories:
    category_data = data[data['distance_category'] == category]
    if len(category_data) > 0:
        category_mode_counts = category_data['chosen_mode'].value_counts()
        category_mode_shares = category_mode_counts / len(category_data) * 100
        mode_share_by_distance[category] = category_mode_shares
        
        print(f"\n{category} ({len(category_data):,} trips):")
        for mode, share in category_mode_shares.items():
            print(f"  {mode}: {share:.1f}% ({category_mode_counts[mode]:,} trips)")

# Focus on longer distances (50km+)
print("\n" + "=" * 60)
print("FOCUS: LONGER DISTANCES (50km+)")
print("=" * 60)

long_distance_data = data[data['trip_length'] >= 50000]  # 50km+
print(f"Total trips 50km+: {len(long_distance_data):,}")

if len(long_distance_data) > 0:
    long_mode_counts = long_distance_data['chosen_mode'].value_counts()
    long_mode_shares = long_mode_counts / len(long_distance_data) * 100
    
    print("\nMode Share for Trips 50km+:")
    for mode, share in long_mode_shares.items():
        print(f"  {mode}: {share:.1f}% ({long_mode_counts[mode]:,} trips)")
    
    # UAM vs Car vs PT comparison for long distances
    print(f"\nUAM vs Traditional Modes (50km+):")
    uam_share = long_mode_shares.get('UAM', 0)
    car_share = long_mode_shares.get('Car', 0)
    pt_share = long_mode_shares.get('Public Transport', 0)
    
    print(f"  UAM: {uam_share:.1f}%")
    print(f"  Car: {car_share:.1f}%")
    print(f"  Public Transport: {pt_share:.1f}%")
    print(f"  Traditional Modes Combined: {car_share + pt_share:.1f}%")
    
    # UAM advantage ratio
    if uam_share > 0:
        traditional_combined = car_share + pt_share
        if traditional_combined > 0:
            uam_ratio = uam_share / traditional_combined
            print(f"  UAM vs Traditional Ratio: 1:{uam_ratio:.2f}")

# Create visualization for mode share by distance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Mode share by distance category
distance_data = []
modes = ['Car', 'Public Transport', 'UAM']
colors = ['#ff9999', '#66b3ff', '#99ff99']

x_pos = np.arange(len(distance_categories))
width = 0.25

for i, mode in enumerate(modes):
    mode_shares = []
    for category in distance_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            mode_shares.append(mode_share_by_distance[category][mode])
        else:
            mode_shares.append(0)
    
    bars = ax1.bar(x_pos + i*width, mode_shares, width, label=mode, color=colors[i])
    
    # Add percentage labels on bars
    for bar, share in zip(bars, mode_shares):
        if share > 0:  # Only add label if share > 0
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{share:.1f}%', ha='center', va='bottom', fontsize=10)

ax1.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mode Share (%)', fontsize=12, fontweight='bold')
ax1.set_title('Mode Share by Trip Distance', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(distance_categories, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Focus on longer distances
if len(long_distance_data) > 0:
    ax2.pie(long_mode_shares.values, labels=long_mode_shares.index, autopct='%1.1f%%',
            colors=colors[:len(long_mode_shares)], startangle=90)
    ax2.set_title('Mode Share for Long Distance Trips (50km+)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/mode_share_by_distance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Create visualization for trip counts by distance (not percentages)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Calculate trip counts by distance category and mode
distance_trip_counts = {}
for category in distance_categories:
    category_data = data[data['distance_category'] == category]
    if len(category_data) > 0:
        category_mode_counts = category_data['chosen_mode'].value_counts()
        distance_trip_counts[category] = category_mode_counts

# Plot trip counts by distance category
modes = ['Car', 'Public Transport', 'UAM']
colors = ['#ff9999', '#66b3ff', '#99ff99']

x_pos = np.arange(len(distance_categories))
width = 0.25

for i, mode in enumerate(modes):
    mode_counts = []
    for category in distance_categories:
        if category in distance_trip_counts and mode in distance_trip_counts[category]:
            mode_counts.append(distance_trip_counts[category][mode])
        else:
            mode_counts.append(0)
    
    bars = ax.bar(x_pos + i*width, mode_counts, width, label=mode, color=colors[i])
    
    # Add value labels on bars
    for bar, count in zip(bars, mode_counts):
        if count > 0:  # Only add label if count > 0
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(count):,}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Distance Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Trips', fontsize=12, fontweight='bold')
ax.set_title('Number of Trips by Mode and Distance Category', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(distance_categories, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/mode_share_trips_by_distance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)
print("Files generated:")
print("mode_share_bar_chart.png")
print("mode_share_percentage_chart.png")
print("mode_share_results.csv")
print("mode_share_by_distance.png")
print("mode_share_trips_by_distance.png")
print("=" * 60)
