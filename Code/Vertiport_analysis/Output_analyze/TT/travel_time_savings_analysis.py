import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
print("Loading data...")
data = pd.read_csv(
    '/Result/Vertiport_analysis/Probability_clustering/Weighting/5km_radius_LightGBM_synthetic_population_predictions_weights.csv',
    low_memory=False)

print(f"Loaded {len(data):,} trips")

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

# Add distance category
data['distance_category'] = data['trip_length'].apply(categorize_distance)

# Calculate travel time savings
data['uam_vs_car_savings'] = data['autos_TT'] - data['travel_time_Uam']
data['uam_vs_pt_savings'] = data['PT_TT'] - data['travel_time_Uam']

print("\n" + "="*80)
print("TRAVEL TIME SAVINGS ANALYSIS")
print("="*80)

# Overall statistics
print(f"\nOVERALL TRAVEL TIME SAVINGS:")
print(f"UAM vs Car - Mean savings: {data['uam_vs_car_savings'].mean():.1f} minutes")
print(f"UAM vs Car - Median savings: {data['uam_vs_car_savings'].median():.1f} minutes")
print(f"UAM vs PT - Mean savings: {data['uam_vs_pt_savings'].mean():.1f} minutes")
print(f"UAM vs PT - Median savings: {data['uam_vs_pt_savings'].median():.1f} minutes")

# Analysis by distance category
distance_categories = ['Short (< 20km)', 'Medium (20-50km)', 'Long (50-100km)', 'Very Long (>100km)']

print(f"\n" + "="*80)
print("TRAVEL TIME SAVINGS BY DISTANCE CATEGORY")
print("="*80)

results_by_distance = {}

for category in distance_categories:
    category_data = data[data['distance_category'] == category]
    if len(category_data) > 0:
        uam_vs_car_mean = category_data['uam_vs_car_savings'].mean()
        uam_vs_car_median = category_data['uam_vs_car_savings'].median()
        uam_vs_pt_mean = category_data['uam_vs_pt_savings'].mean()
        uam_vs_pt_median = category_data['uam_vs_pt_savings'].median()
        
        results_by_distance[category] = {
            'uam_vs_car_mean': uam_vs_car_mean,
            'uam_vs_car_median': uam_vs_car_median,
            'uam_vs_pt_mean': uam_vs_pt_mean,
            'uam_vs_pt_median': uam_vs_pt_median,
            'count': len(category_data)
        }
        
        print(f"\n{category} ({len(category_data):,} trips):")
        print(f"  UAM vs Car - Mean savings: {uam_vs_car_mean:.1f} minutes")
        print(f"  UAM vs Car - Median savings: {uam_vs_car_median:.1f} minutes")
        print(f"  UAM vs PT - Mean savings: {uam_vs_pt_mean:.1f} minutes")
        print(f"  UAM vs PT - Median savings: {uam_vs_pt_median:.1f} minutes")

# Focus on UAM trips only
print(f"\n" + "="*80)
print("TRAVEL TIME SAVINGS FOR UAM TRIPS ONLY")
print("="*80)

# Filter for UAM trips (where UAM has highest probability)
uam_trips = data[data['prob_mode_Autonomous Flying Taxi'] > data[['prob_mode_Car', 'prob_mode_Public Transport']].max(axis=1)]
print(f"Total UAM trips: {len(uam_trips):,}")

for category in distance_categories:
    category_uam = uam_trips[uam_trips['distance_category'] == category]
    if len(category_uam) > 0:
        uam_vs_car_mean = category_uam['uam_vs_car_savings'].mean()
        uam_vs_pt_mean = category_uam['uam_vs_pt_savings'].mean()
        
        print(f"\n{category} UAM trips ({len(category_uam):,} trips):")
        print(f"  UAM vs Car - Mean savings: {uam_vs_car_mean:.1f} minutes")
        print(f"  UAM vs PT - Mean savings: {uam_vs_pt_mean:.1f} minutes")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: UAM vs Car savings by distance
categories = list(results_by_distance.keys())
uam_vs_car_means = [results_by_distance[cat]['uam_vs_car_mean'] for cat in categories]
uam_vs_pt_means = [results_by_distance[cat]['uam_vs_pt_mean'] for cat in categories]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, uam_vs_car_means, width, label='UAM vs Car', color='#ff7f0e', alpha=0.8)
bars2 = ax1.bar(x + width/2, uam_vs_pt_means, width, label='UAM vs PT', color='#1f77b4', alpha=0.8)

ax1.set_xlabel('Distance Category')
ax1.set_ylabel('Time Savings (minutes)')
ax1.set_title('UAM Travel Time Savings by Distance Category')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

# Plot 2: Distribution of UAM vs PT savings
ax2.hist(data['uam_vs_pt_savings'], bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
ax2.set_xlabel('UAM vs PT Time Savings (minutes)')
ax2.set_ylabel('Number of Trips')
ax2.set_title('Distribution of UAM vs Public Transport Time Savings')
ax2.axvline(data['uam_vs_pt_savings'].mean(), color='red', linestyle='--', 
           label=f'Mean: {data["uam_vs_pt_savings"].mean():.1f} min')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Distribution of UAM vs Car savings
ax3.hist(data['uam_vs_car_savings'], bins=50, alpha=0.7, color='#ff7f0e', edgecolor='black')
ax3.set_xlabel('UAM vs Car Time Savings (minutes)')
ax3.set_ylabel('Number of Trips')
ax3.set_title('Distribution of UAM vs Car Time Savings')
ax3.axvline(data['uam_vs_car_savings'].mean(), color='red', linestyle='--',
           label=f'Mean: {data["uam_vs_car_savings"].mean():.1f} min')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Scatter plot - Trip length vs Time savings
ax4.scatter(data['trip_length']/1000, data['uam_vs_pt_savings'], alpha=0.5, s=1, color='#1f77b4', label='UAM vs PT')
ax4.scatter(data['trip_length']/1000, data['uam_vs_car_savings'], alpha=0.5, s=1, color='#ff7f0e', label='UAM vs Car')
ax4.set_xlabel('Trip Length (km)')
ax4.set_ylabel('Time Savings (minutes)')
ax4.set_title('Time Savings vs Trip Length')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/travel_time_savings_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()

# Save results to CSV
results_df = pd.DataFrame([
    {
        'Distance_Category': category,
        'Trip_Count': results_by_distance[category]['count'],
        'UAM_vs_Car_Mean_Savings': results_by_distance[category]['uam_vs_car_mean'],
        'UAM_vs_Car_Median_Savings': results_by_distance[category]['uam_vs_car_median'],
        'UAM_vs_PT_Mean_Savings': results_by_distance[category]['uam_vs_pt_mean'],
        'UAM_vs_PT_Median_Savings': results_by_distance[category]['uam_vs_pt_median']
    }
    for category in distance_categories
])

results_df.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/travel_time_savings_results.csv', index=False)

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("Files generated:")
print("- travel_time_savings_analysis.png")
print("- travel_time_savings_results.csv")
print("="*80)
