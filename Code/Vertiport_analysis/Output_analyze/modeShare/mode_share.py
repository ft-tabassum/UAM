import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv(
    'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/cost_LightGBM_synthetic_population_predictions_weights.csv')

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
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/11',
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
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/mode_share_percentage_chart.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Save results to CSV
mode_share_results = pd.DataFrame({
    'Mode': mode_counts.index,
    'Count': mode_counts.values,
    'Percentage': mode_shares.values
})
mode_share_results.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/12',
                         index=False)

print("\n" + "=" * 60)
print("MODE SHARE BY TRIP DISTANCE")
print("=" * 60)

# Define distance categories (updated to 20-50, 50-100, 100-150 km)
def categorize_distance(trip_length):
    if trip_length < 20000:  # < 20 km
        return '< 20'
    elif trip_length < 50000:  # 20-50 km
        return '20-50'
    elif trip_length < 100000:  # 50-100 km
        return '50-100'
    elif trip_length < 150000:  # 100-150 km
        return '100-150'
    else:  # >= 150 km
        return '150+'

# Add distance category to data
data['distance_category'] = data['trip_length'].apply(categorize_distance)

# Analyze mode share by distance category
print("\nMode Share by Distance Category:")
print("-" * 50)

distance_categories = ['< 20', '20-50', '50-100', '100-150', '150+']
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

#  longer distances (50km+)
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
ax1.set_xticklabels(distance_categories)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Focus on longer distances
if len(long_distance_data) > 0:
    ax2.pie(long_mode_shares.values, labels=long_mode_shares.index, autopct='%1.1f%%',
            colors=colors[:len(long_mode_shares)], startangle=90)
    ax2.set_title('Mode Share for Long Distance Trips (50km+)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/13',
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

ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Trips', fontsize=12, fontweight='bold')
ax.set_title('Number of Trips by Mode and Distance Category', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(distance_categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/1',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# ADDITIONAL ADVANCED VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 60)
print("CREATING ADVANCED VISUALIZATIONS")
print("=" * 60)

# Visualization 1: Stacked Bar Chart (100% stacked)
fig, ax = plt.subplots(figsize=(12, 8))

modes = ['Car', 'Public Transport', 'UAM']
colors = ['#e74c3c', '#3498db', '#2ecc71']

# Prepare data for 100% stacked bar
mode_shares_by_dist = []
for mode in modes:
    shares = []
    for category in distance_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            shares.append(mode_share_by_distance[category][mode])
        else:
            shares.append(0)
    mode_shares_by_dist.append(shares)

# Create stacked bars
x_pos = np.arange(len(distance_categories))
bottom = np.zeros(len(distance_categories))

for i, (mode, shares) in enumerate(zip(modes, mode_shares_by_dist)):
    bars = ax.bar(x_pos, shares, label=mode, color=colors[i], bottom=bottom)
    
    # Add percentage labels
    for j, (bar, share) in enumerate(zip(bars, shares)):
        if share > 3:  # Only show label if share > 3%
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   bottom[j] + height/2,
                   f'{share:.1f}%', 
                   ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='white')
    
    bottom += np.array(shares)

ax.set_xlabel('Distance (km)', fontsize=13, fontweight='bold')
ax.set_ylabel('Mode Share (%)', fontsize=13, fontweight='bold')
ax.set_title('Mode Share Distribution by Trip Distance\n(100% Stacked Bar Chart)', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(distance_categories, fontsize=11)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/mode_share_stacked_by_distance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Created: mode_share_stacked_by_distance.png")
plt.close()

# Visualization 2: Line Chart showing mode share trends
fig, ax = plt.subplots(figsize=(14, 8))

for i, mode in enumerate(modes):
    shares = []
    for category in distance_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            shares.append(mode_share_by_distance[category][mode])
        else:
            shares.append(0)
    
    ax.plot(distance_categories, shares, marker='o', linewidth=3, 
           markersize=10, label=mode, color=colors[i])
    
    # Add value labels on points
    for j, (cat, share) in enumerate(zip(distance_categories, shares)):
        if share > 0:
            ax.annotate(f'{share:.1f}%', 
                       xy=(j, share), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center', 
                       fontsize=10, 
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=colors[i], 
                                alpha=0.3))

ax.set_xlabel('Distance (km)', fontsize=13, fontweight='bold')
ax.set_ylabel('Mode Share (%)', fontsize=13, fontweight='bold')
ax.set_title('Mode Share Trend Across Distance Categories\n(How Mode Choice Changes with Trip Distance)', 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, max([max(mode_shares_by_dist[i]) for i in range(len(modes))]) * 1.2)

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/mode_share_trend_line_by_distance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Created: mode_share_trend_line_by_distance.png")
plt.close()

# Visualization 3: Horizontal Bar Chart (easier to read mode names)
fig, ax = plt.subplots(figsize=(14, 10))

y_pos = np.arange(len(distance_categories))
width = 0.25

for i, mode in enumerate(modes):
    shares = []
    for category in distance_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            shares.append(mode_share_by_distance[category][mode])
        else:
            shares.append(0)
    
    bars = ax.barh(y_pos + i*width, shares, width, label=mode, color=colors[i], alpha=0.85)
    
    # Add value labels
    for bar, share in zip(bars, shares):
        if share > 0:
            width_val = bar.get_width()
            ax.text(width_val + 1, bar.get_y() + bar.get_height()/2.,
                   f'{share:.1f}%', ha='left', va='center', 
                   fontsize=10, fontweight='bold')

ax.set_ylabel('Distance (km)', fontsize=13, fontweight='bold')
ax.set_xlabel('Mode Share (%)', fontsize=13, fontweight='bold')
ax.set_title('Mode Share Comparison by Distance Category\n(Horizontal View)', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_yticks(y_pos + width)
ax.set_yticklabels(distance_categories, fontsize=11)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/mode_share_horizontal_by_distance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Created: mode_share_horizontal_by_distance.png")
plt.close()

# Visualization 4: Heatmap showing mode share intensity
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare heatmap data
heatmap_data = []
for mode in modes:
    shares = []
    for category in distance_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            shares.append(mode_share_by_distance[category][mode])
        else:
            shares.append(0)
    heatmap_data.append(shares)

heatmap_array = np.array(heatmap_data)

# Create heatmap
im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

# Set ticks and labels
ax.set_xticks(np.arange(len(distance_categories)))
ax.set_yticks(np.arange(len(modes)))
ax.set_xticklabels(distance_categories, fontsize=11)
ax.set_yticklabels(modes, fontsize=11)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mode Share (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(modes)):
    for j in range(len(distance_categories)):
        text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}%',
                      ha="center", va="center", color="black" if heatmap_array[i, j] < 50 else "white",
                      fontsize=11, fontweight='bold')

ax.set_xlabel('Distance (km)', fontsize=13, fontweight='bold')
ax.set_ylabel('Transportation Mode', fontsize=13, fontweight='bold')
ax.set_title('Mode Share Intensity Heatmap by Distance\n(Darker = Higher Share)', 
            fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/mode_share_heatmap_by_distance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Created: mode_share_heatmap_by_distance.png")
plt.close()

# Visualization 5: Focused comparison on key distance ranges (20-50, 50-100, 100-150)
focus_categories = ['20-50', '50-100', '100-150']
fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(focus_categories))
width = 0.25

# Custom colors for this graph - Car: Blue, PT: Green, UAM: Orange
focus_colors = ['#3498db', '#2ecc71', '#ff8c42']  # Blue, Green, Orange

for i, mode in enumerate(modes):
    shares = []
    for category in focus_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            shares.append(mode_share_by_distance[category][mode])
        else:
            shares.append(0)
    
    bars = ax.bar(x_pos + i*width, shares, width, label=mode, color=focus_colors[i], alpha=0.85)
    
    # Add percentage labels on bars - MUCH LARGER
    for bar, share in zip(bars, shares):
        if share > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{share:.1f}%', ha='center', va='bottom', 
                   fontsize=18, fontweight='bold')

ax.set_xlabel('Distance (km)', fontsize=20, fontweight='bold')
ax.set_ylabel('Mode Share (%)', fontsize=20, fontweight='bold')
# ax.set_title('Mode Share Comparison: Focus on 20-150 km Range\n(Key Distance Categories for UAM Competitiveness)', 
#             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(focus_categories, fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=16, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/mode_share_focus_20_150km.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Created: mode_share_focus_20_150km.png")
plt.close()

# Save detailed mode share table
mode_share_table = pd.DataFrame()
mode_share_table['Distance (km)'] = distance_categories

for mode in modes:
    shares = []
    for category in distance_categories:
        if category in mode_share_by_distance and mode in mode_share_by_distance[category]:
            shares.append(f"{mode_share_by_distance[category][mode]:.2f}%")
        else:
            shares.append("0.00%")
    mode_share_table[mode] = shares

mode_share_table.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario_base0_perkm3/mode_share_by_distance_table.csv',
                       index=False)
print("Created: mode_share_by_distance_table.csv")

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)
print("Files generated:")
print("  1. mode_share_bar_chart.png")
print("  2. mode_share_percentage_chart.png")
print("  3. mode_share_results.csv")
print("  4. mode_share_by_distance.png")
print("  5. mode_share_trips_by_distance.png")
print("  6. mode_share_stacked_by_distance.png (NEW)")
print("  7. mode_share_trend_line_by_distance.png (NEW)")
print("  8. mode_share_horizontal_by_distance.png (NEW)")
print("  9. mode_share_heatmap_by_distance.png (NEW)")
print(" 10. mode_share_focus_20_150km.png (NEW)")
print(" 11. mode_share_by_distance_table.csv (NEW)")
print("=" * 60)
print("\nRECOMMENDATIONS FOR PRESENTATION:")
print("-" * 60)
print("1. BEST for showing trends: mode_share_trend_line_by_distance.png")
print("   - Clearly shows how each mode's share changes with distance")
print("   - Easy to see UAM's pattern across categories")
print("\n2. BEST for composition: mode_share_stacked_by_distance.png")
print("   - Shows 100% composition at each distance")
print("   - Easy to compare relative importance")
print("\n3. BEST for detailed comparison: mode_share_heatmap_by_distance.png")
print("   - Color intensity shows patterns at a glance")
print("   - Good for presentations and papers")
print("\n4. BEST for focused analysis: mode_share_focus_20_150km.png")
print("   - Focuses on key 20-150 km range")
print("   - Emphasizes UAM's competitive distances")
print("\n5. MOST READABLE: mode_share_horizontal_by_distance.png")
print("   - Horizontal layout easier to read mode names")
print("   - Good for detailed reports")
print("=" * 60)
