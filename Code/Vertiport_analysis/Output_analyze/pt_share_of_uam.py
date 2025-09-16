import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

print("=" * 80)
print("PUBLIC TRANSPORT SHARE OF UAM ANALYSIS")
print("=" * 80)

# Step 1: Filter for UAM trips only (where UAM is the chosen mode)
# A trip is considered UAM if prob_AFT is highest among all modes
def determine_mode(row):
    if row['prob_mode_Car'] > row['prob_mode_Public Transport'] and row['prob_mode_Car'] > row['prob_mode_Autonomous Flying Taxi']:
        return 'Car'
    elif row['prob_mode_Public Transport'] > row['prob_mode_Car'] and row['prob_mode_Public Transport'] > row['prob_mode_Autonomous Flying Taxi']:
        return 'Public Transport'
    elif row['prob_mode_Autonomous Flying Taxi'] > row['prob_mode_Car'] and row['prob_mode_Autonomous Flying Taxi'] > row['prob_mode_Public Transport']:
        return 'UAM'
    else:
        return 'Tie'

# Apply the function to determine mode for each trip
data['chosen_mode'] = data.apply(determine_mode, axis=1)

# Filter for UAM trips only
uam_trips = data[data['chosen_mode'] == 'UAM'].copy()

print(f"Total number of trips in dataset: {len(data):,}")
print(f"Total number of UAM trips: {len(uam_trips):,}")
print(f"UAM mode share: {len(uam_trips)/len(data)*100:.2f}%")

# Step 2: Calculate Public Transport Share of UAM
# UAM replaces PT when prob_AFT > prob_PT for UAM trips
print("\n" + "=" * 60)
print("CALCULATING PT SHARE OF UAM")
print("=" * 60)

# For UAM trips, check if prob_AFT > prob_PT AND PT was the second choice (not Car)
# This indicates that UAM is replacing PT for those trips
uam_trips['uam_replaces_pt'] = (
    (uam_trips['prob_mode_Autonomous Flying Taxi'] > uam_trips['prob_mode_Public Transport']) &
    (uam_trips['prob_mode_Public Transport'] > uam_trips['prob_mode_Car'])
)

# Count UAM trips that replace PT
uam_trips_replacing_pt = uam_trips[uam_trips['uam_replaces_pt'] == True]
total_uam_trips = len(uam_trips)

print(f"UAM trips where prob_AFT > prob_PT AND PT was second choice: {len(uam_trips_replacing_pt):,}")
print(f"Total UAM trips: {total_uam_trips:,}")

# Calculate PT Share of UAM
if total_uam_trips > 0:
    pt_share_of_uam = (len(uam_trips_replacing_pt) / total_uam_trips) * 100
    print(f"\nPublic Transport Share of UAM: {pt_share_of_uam:.2f}%")
else:
    pt_share_of_uam = 0
    print(f"\nPublic Transport Share of UAM: {pt_share_of_uam:.2f}%")

# Step 3: Detailed Analysis
print("\n" + "=" * 60)
print("DETAILED ANALYSIS")
print("=" * 60)

# Probability comparison for UAM trips
print(f"\nProbability Statistics for UAM Trips:")
print(f"UAM probability range: {uam_trips['prob_mode_Autonomous Flying Taxi'].min():.4f} - {uam_trips['prob_mode_Autonomous Flying Taxi'].max():.4f}")
print(f"PT probability range: {uam_trips['prob_mode_Public Transport'].min():.4f} - {uam_trips['prob_mode_Public Transport'].max():.4f}")
print(f"Car probability range: {uam_trips['prob_mode_Car'].min():.4f} - {uam_trips['prob_mode_Car'].max():.4f}")

print(f"\nAverage probabilities for UAM trips:")
print(f"UAM: {uam_trips['prob_mode_Autonomous Flying Taxi'].mean():.4f}")
print(f"PT: {uam_trips['prob_mode_Public Transport'].mean():.4f}")
print(f"Car: {uam_trips['prob_mode_Car'].mean():.4f}")

# Analysis by probability difference
uam_trips['prob_diff_aft_pt'] = uam_trips['prob_mode_Autonomous Flying Taxi'] - uam_trips['prob_mode_Public Transport']
uam_trips['prob_diff_aft_car'] = uam_trips['prob_mode_Autonomous Flying Taxi'] - uam_trips['prob_mode_Car']

print(f"\nProbability differences for UAM trips:")
print(f"UAM vs PT difference range: {uam_trips['prob_diff_aft_pt'].min():.4f} to {uam_trips['prob_diff_aft_pt'].max():.4f}")
print(f"UAM vs Car difference range: {uam_trips['prob_diff_aft_car'].min():.4f} to {uam_trips['prob_diff_aft_car'].max():.4f}")
print(f"Average UAM vs PT difference: {uam_trips['prob_diff_aft_pt'].mean():.4f}")
print(f"Average UAM vs Car difference: {uam_trips['prob_diff_aft_car'].mean():.4f}")

# Count how many UAM trips replace each mode
uam_replacing_car = len(uam_trips[uam_trips['prob_diff_aft_car'] > 0])
uam_replacing_pt = len(uam_trips[uam_trips['prob_diff_aft_pt'] > 0])

# Determine second choice for each UAM trip
def get_second_choice(row):
    if row['prob_mode_Public Transport'] > row['prob_mode_Car']:
        return 'Public Transport'
    else:
        return 'Car'

uam_trips['second_choice'] = uam_trips.apply(get_second_choice, axis=1)
second_choice_counts = uam_trips['second_choice'].value_counts()

print(f"\nUAM replacement analysis:")
print(f"UAM trips replacing Car (prob_AFT > prob_Car): {uam_replacing_car:,}")
print(f"UAM trips replacing PT (prob_AFT > prob_PT): {uam_replacing_pt:,}")

print(f"\nSecond choice analysis for UAM trips:")
print(f"UAM trips where PT was second choice: {second_choice_counts.get('Public Transport', 0):,}")
print(f"UAM trips where Car was second choice: {second_choice_counts.get('Car', 0):,}")

# Step 4: Distance-based analysis
print("\n" + "=" * 60)
print("DISTANCE-BASED ANALYSIS")
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

# Add distance category to UAM trips
uam_trips['distance_category'] = uam_trips['trip_length'].apply(categorize_distance)

distance_categories = ['Short (< 20km)', 'Medium (20-50km)', 'Long (50-100km)', 'Very Long (>100km)']
pt_share_by_distance = {}

print(f"\nPT Share of UAM by Distance Category:")
print("-" * 50)

for category in distance_categories:
    category_uam = uam_trips[uam_trips['distance_category'] == category]
    if len(category_uam) > 0:
        category_replacing_pt = len(category_uam[category_uam['uam_replaces_pt'] == True])
        category_pt_share = (category_replacing_pt / len(category_uam)) * 100
        pt_share_by_distance[category] = category_pt_share
        
        print(f"{category} ({len(category_uam):,} UAM trips):")
        print(f"  UAM trips replacing PT: {category_replacing_pt:,}")
        print(f"  PT Share of UAM: {category_pt_share:.2f}%")

# Step 5: Create Visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Visualization 1: PT Share of UAM overview
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Overall PT Share of UAM
labels = ['UAM replacing PT', 'UAM not replacing PT']
sizes = [len(uam_trips_replacing_pt), total_uam_trips - len(uam_trips_replacing_pt)]
colors = ['#ff6b6b', '#4ecdc4']

wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title(f'Public Transport Share of UAM\n({pt_share_of_uam:.2f}% of UAM trips replace PT)', 
              fontsize=14, fontweight='bold')

# Plot 2: PT Share by Distance Category
if pt_share_by_distance:
    categories = list(pt_share_by_distance.keys())
    shares = list(pt_share_by_distance.values())
    
    bars = ax2.bar(categories, shares, color='#ff9999', alpha=0.7)
    ax2.set_title('PT Share of UAM by Distance Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PT Share (%)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, share in zip(bars, shares):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{share:.1f}%', ha='center', va='bottom', fontsize=10)



plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/pt_share_of_uam_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Step 6: Save detailed results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Create summary results DataFrame
summary_results = pd.DataFrame({
    'Metric': [
        'Total trips in dataset',
        'Total UAM trips',
        'UAM trips replacing PT',
        'PT Share of UAM (%)',
        'UAM trips replacing Car',
        'Average UAM probability',
        'Average PT probability',
        'Average Car probability'
    ],
    'Value': [
        len(data),
        total_uam_trips,
        len(uam_trips_replacing_pt),
        pt_share_of_uam,
        uam_replacing_car,
        uam_trips['prob_mode_Autonomous Flying Taxi'].mean(),
        uam_trips['prob_mode_Public Transport'].mean(),
        uam_trips['prob_mode_Car'].mean()
    ]
})

summary_results.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/pt_share_of_uam_summary.csv', 
                       index=False)

# Save distance-based analysis
distance_analysis = []
for category in distance_categories:
    if category in pt_share_by_distance:
        category_uam = uam_trips[uam_trips['distance_category'] == category]
        category_replacing_pt = len(category_uam[category_uam['uam_replaces_pt'] == True])
        distance_analysis.append({
            'Distance_Category': category,
            'Total_UAM_Trips': len(category_uam),
            'UAM_Replacing_PT': category_replacing_pt,
            'PT_Share_of_UAM': pt_share_by_distance[category]
        })

distance_df = pd.DataFrame(distance_analysis)
distance_df.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/pt_share_by_distance.csv', 
                   index=False)

# Save detailed UAM trip analysis
uam_analysis = uam_trips[['trip_length', 'distance_category', 
                         'prob_mode_Autonomous Flying Taxi', 
                         'prob_mode_Public Transport', 
                         'prob_mode_Car', 
                         'prob_diff_aft_pt', 
                         'prob_diff_aft_car', 
                         'uam_replaces_pt']].copy()

uam_analysis.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/uam_detailed_analysis.csv', 
                    index=False)

print("Files generated:")
print("- pt_share_of_uam_analysis.png")
print("- pt_share_of_uam_summary.csv")
print("- pt_share_by_distance.csv")
print("- uam_detailed_analysis.csv")

print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"Public Transport Share of UAM: {pt_share_of_uam:.2f}%")
print(f"This means {pt_share_of_uam:.2f}% of UAM trips are replacing public transport")
print(f"Out of {total_uam_trips:,} total UAM trips, {len(uam_trips_replacing_pt):,} replace PT")
print("=" * 80)
