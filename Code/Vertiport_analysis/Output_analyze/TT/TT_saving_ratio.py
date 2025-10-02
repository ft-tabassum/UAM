import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

data = pd.read_csv(
    '/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

# Calculate rtts for UAM
# rtts = 1 - (UAM_TT / groundbased, motorized mode (i.e., car or PT) TT
data['rtts'] = 1 - (data['travel_time_Uam'] / data['autos_TT'])  # autos

# Calculate Weighted Travel Time Savings for UAM
# Weighted TT Savings for UAM = rtts * prob_AFT (probability of using UAM)
data['Weighted_TT_Savings'] = data['rtts'] * data['prob_mode_Autonomous Flying Taxi']

# Filter data for positive RTTs only (UAM is faster than ground transport)
positive_rtts_data = data[data['rtts'] > 0].copy()

print("=" * 80)
print("RELATIVE TRAVEL TIME SAVINGS (RTTs) ANALYSIS - POSITIVE RTTs ONLY")
print("=" * 80)
print(f"Total trips analyzed: {len(data):,}")
print(f"Trips with positive RTTs (UAM faster): {len(positive_rtts_data):,}")
print(f"Percentage of trips where UAM is faster: {len(positive_rtts_data)/len(data)*100:.2f}%")
print("=" * 80)

if len(positive_rtts_data) > 0:
    # Calculate Average RTTs for positive RTTs only
    # Simple average (unweighted) for positive RTTs
    simple_avg_positive_rtts = positive_rtts_data['rtts'].mean()
    
    # Weighted average using UAM probability as weights for positive RTTs
    weighted_avg_positive_rtts = (positive_rtts_data['rtts'] * positive_rtts_data['prob_mode_Autonomous Flying Taxi']).sum() / positive_rtts_data['prob_mode_Autonomous Flying Taxi'].sum()
    
    # Additional statistics for positive RTTs
    median_positive_rtts = positive_rtts_data['rtts'].median()
    std_positive_rtts = positive_rtts_data['rtts'].std()
    
    # Print results for positive RTTs
    print("\nPOSITIVE RTTs ANALYSIS (UAM is FASTER than ground transport):")
    print("-" * 60)
    print(f"Simple Average RTTs (positive only): {simple_avg_positive_rtts:.4f} ({simple_avg_positive_rtts*100:.2f}%)")
    print(f"Weighted Average RTTs (positive only): {weighted_avg_positive_rtts:.4f} ({weighted_avg_positive_rtts*100:.2f}%)")
    print(f"Median RTTs (positive only): {median_positive_rtts:.4f} ({median_positive_rtts*100:.2f}%)")
    print(f"Standard Deviation (positive only): {std_positive_rtts:.4f}")
    print(f"Trips with positive RTTs: {len(positive_rtts_data):,}")
    print("-" * 60)
else:
    print("\nWARNING: No trips found where UAM is faster than ground transport!")
    print("All RTTs values are negative or zero.")
    simple_avg_positive_rtts = 0
    weighted_avg_positive_rtts = 0
    median_positive_rtts = 0
    std_positive_rtts = 0

# Create separate visualizations for positive RTTs analysis

if len(positive_rtts_data) > 0:
    # Visualization 1: Distribution of RTTs (UAM is Faster)
    plt.figure(figsize=(12, 8))
    plt.hist(positive_rtts_data['rtts'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(simple_avg_positive_rtts, color='red', linestyle='--', linewidth=2, label=f'Simple Avg: {simple_avg_positive_rtts:.3f}')
    plt.axvline(weighted_avg_positive_rtts, color='blue', linestyle='--', linewidth=2, label=f'Weighted Avg: {weighted_avg_positive_rtts:.3f}')
    plt.xlabel('Relative Travel Time Savings (RTTs)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RTTs (UAM is Faster)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/rtts_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 2: UAM Probability vs RTTs with Statistics
    plt.figure(figsize=(12, 8))
    plt.scatter(positive_rtts_data['prob_mode_Autonomous Flying Taxi'], positive_rtts_data['rtts'], alpha=0.6, s=2, color='green')
    
    # Add correlation coefficient and statistics
    correlation = positive_rtts_data['prob_mode_Autonomous Flying Taxi'].corr(positive_rtts_data['rtts'])
    
    # Add text box with statistics
    stats_text = f'Correlation: {correlation:.3f}\nTotal trips: {len(positive_rtts_data):,}\nAvg RTTs: {simple_avg_positive_rtts:.3f}\nAvg UAM Prob: {positive_rtts_data["prob_mode_Autonomous Flying Taxi"].mean():.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('UAM Probability')
    plt.ylabel('RTTs')
    plt.title('UAM Probability vs RTTs (UAM is Faster)')
    plt.grid(True, alpha=0.3)
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/uam_probability_vs_rtts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 3: Summary Statistics
    plt.figure(figsize=(8, 6))
    stats_data = {
        'Simple Avg': simple_avg_positive_rtts,
        'Weighted Avg': weighted_avg_positive_rtts,
        'Median': median_positive_rtts
    }
    bars = plt.bar(stats_data.keys(), stats_data.values(), color=['red', 'blue', 'orange'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, stats_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('RTTs Value')
    plt.title('RTTs Summary Statistics (UAM is Faster)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/rtts_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("No positive RTTs found - skipping positive RTTs visualizations")

# Save the results to a new CSV file
data.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/weighted_tt_savings_results.csv', index=False)

# Save summary statistics for positive RTTs
if len(positive_rtts_data) > 0:
    summary_stats = {
        'Metric': ['Simple Average RTTs (Positive)', 'Weighted Average RTTs (Positive)', 'Median RTTs (Positive)', 'Standard Deviation (Positive)', 'Total Positive RTTs Trips', 'Total Trips Analyzed'],
        'Value': [simple_avg_positive_rtts, weighted_avg_positive_rtts, median_positive_rtts, std_positive_rtts, len(positive_rtts_data), len(data)],
        'Percentage': [f"{simple_avg_positive_rtts*100:.2f}%", f"{weighted_avg_positive_rtts*100:.2f}%", f"{median_positive_rtts*100:.2f}%", f"{std_positive_rtts*100:.2f}%", f"{len(positive_rtts_data):,}", f"{len(data):,}"]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/positive_rtts_summary_statistics.csv', index=False)
else:
    print("No positive RTTs data to save in summary statistics")

print(f"\nInitial results saved to:")
print(f"- weighted_tt_savings_results.csv")
print(f"- positive_rtts_summary_statistics.csv")

# Additional Analysis: Trip Length Analysis for Positive RTTs
print("\n" + "=" * 80)
print("TRIP LENGTH ANALYSIS FOR POSITIVE RTTs (UAM is FASTER)")
print("=" * 80)

if len(positive_rtts_data) > 0:
    # Convert trip_length from meters to kilometers for positive RTTs data
    positive_rtts_data['trip_length_km'] = positive_rtts_data['trip_length'] / 1000
    
    # Create trip length bins for positive RTTs
    positive_rtts_data['trip_length_bin'] = pd.cut(positive_rtts_data['trip_length_km'], 
                                                  bins=[0, 5, 10, 15, 20, 25, 30, 50, 100], 
                                                  labels=['0-5km', '5-10km', '10-15km', '15-20km', '20-25km', '25-30km', '30-50km', '50km+'])
    
    # Calculate analysis by trip length for positive RTTs
    positive_trip_length_analysis = positive_rtts_data.groupby('trip_length_bin').agg({
        'Weighted_TT_Savings': 'sum',
        'rtts': 'mean',
        'travel_time_Uam': 'mean',
        'autos_TT': 'mean',
        'trip_id': 'count'
    }).round(4)
    
    # Calculate weighted average RTTs by trip length for positive RTTs
    positive_trip_length_analysis['weighted_avg_rtts'] = (
        positive_rtts_data.groupby('trip_length_bin').apply(
            lambda x: (x['rtts'] * x['prob_mode_Autonomous Flying Taxi']).sum() / x['prob_mode_Autonomous Flying Taxi'].sum()
        )
    ).round(4)
    
    # Rename columns for clarity
    positive_trip_length_analysis.columns = [
        'Total_Weighted_TT_Savings', 
        'Simple_Avg_RTTs', 
        'Avg_UAM_Travel_Time', 
        'Avg_Auto_Travel_Time',
        'Trip_Count',
        'Weighted_Avg_RTTs'
    ]
    
    # Add percentage columns
    positive_trip_length_analysis['Simple_Avg_RTTs_Pct'] = (positive_trip_length_analysis['Simple_Avg_RTTs'] * 100).round(2)
    positive_trip_length_analysis['Weighted_Avg_RTTs_Pct'] = (positive_trip_length_analysis['Weighted_Avg_RTTs'] * 100).round(2)
    
    # Print results for positive RTTs by trip length
    print("\nPositive RTTs Trip Length Analysis:")
    print("-" * 80)
    for bin_name, row in positive_trip_length_analysis.iterrows():
        if row['Trip_Count'] > 0:  # Only show bins with data
            print(f"{bin_name:>8}: {row['Trip_Count']:>8,} trips | "
                  f"Simple RTTs: {row['Simple_Avg_RTTs_Pct']:>7.2f}% | "
                  f"Weighted RTTs: {row['Weighted_Avg_RTTs_Pct']:>7.2f}% | "
                  f"Avg UAM TT: {row['Avg_UAM_Travel_Time']:>6.1f}min | "
                  f"Avg Auto TT: {row['Avg_Auto_Travel_Time']:>6.1f}min")
else:
    print("No positive RTTs data available for trip length analysis")
    positive_trip_length_analysis = pd.DataFrame()

# Create separate visualizations for trip length analysis of positive RTTs

if len(positive_trip_length_analysis) > 0 and len(positive_trip_length_analysis[positive_trip_length_analysis['Trip_Count'] > 0]) > 0:
    
    # Filter out empty bins
    filtered_analysis = positive_trip_length_analysis[positive_trip_length_analysis['Trip_Count'] > 0]
    
    # Visualization 1: Simple Average RTTs by Trip Length
    plt.figure(figsize=(12, 8))
    bars = filtered_analysis['Simple_Avg_RTTs_Pct'].plot(kind='bar', color='red', alpha=0.7)
    plt.title('Simple Average RTTs by Trip Length (UAM is Faster)')
    plt.xlabel('Trip Length (km)')
    plt.ylabel('RTTs (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (idx, value) in enumerate(filtered_analysis['Simple_Avg_RTTs_Pct'].items()):
        plt.text(i, value + 0.2, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/simple_avg_rtts_by_trip_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 2: Weighted Average RTTs by Trip Length
    plt.figure(figsize=(12, 8))
    bars = filtered_analysis['Weighted_Avg_RTTs_Pct'].plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Weighted Average RTTs by Trip Length (UAM is Faster)')
    plt.xlabel('Trip Length (km)')
    plt.ylabel('RTTs (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (idx, value) in enumerate(filtered_analysis['Weighted_Avg_RTTs_Pct'].items()):
        plt.text(i, value + 0.2, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/weighted_avg_rtts_by_trip_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 3: Travel Time Comparison by Trip Length
    plt.figure(figsize=(12, 8))
    x_pos = range(len(filtered_analysis))
    uam_bars = plt.bar([x - 0.2 for x in x_pos], filtered_analysis['Avg_UAM_Travel_Time'], 
            width=0.4, label='UAM', color='blue', alpha=0.7)
    auto_bars = plt.bar([x + 0.2 for x in x_pos], filtered_analysis['Avg_Auto_Travel_Time'], 
            width=0.4, label='Auto', color='orange', alpha=0.7)
    
    # Add value labels on bars
    for i, (uam_val, auto_val) in enumerate(zip(filtered_analysis['Avg_UAM_Travel_Time'], filtered_analysis['Avg_Auto_Travel_Time'])):
        plt.text(i - 0.2, uam_val + 0.5, f'{uam_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.text(i + 0.2, auto_val + 0.5, f'{auto_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.title('Average Travel Time by Trip Length (UAM is Faster)')
    plt.xlabel('Trip Length (km)')
    plt.ylabel('Travel Time (minutes)')
    plt.xticks(x_pos, filtered_analysis.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/travel_time_by_trip_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 4: Trip Count by Length
    plt.figure(figsize=(12, 8))
    bars = filtered_analysis['Trip_Count'].plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Number of Trips by Length (UAM is Faster)')
    plt.xlabel('Trip Length (km)')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, value) in enumerate(filtered_analysis['Trip_Count'].items()):
        plt.text(i, value + max(filtered_analysis['Trip_Count']) * 0.01, f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/trip_count_by_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 5: Total Weighted TT Savings by Trip Length
    plt.figure(figsize=(12, 8))
    bars = filtered_analysis['Total_Weighted_TT_Savings'].plot(kind='bar', color='teal', alpha=0.7)
    plt.title('Total Weighted TT Savings by Trip Length (UAM is Faster)')
    plt.xlabel('Trip Length (km)')
    plt.ylabel('Total Weighted TT Savings')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, value) in enumerate(filtered_analysis['Total_Weighted_TT_Savings'].items()):
        plt.text(i, value + max(filtered_analysis['Total_Weighted_TT_Savings']) * 0.01, f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/weighted_savings_by_trip_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # Save trip length analysis for positive RTTs
    positive_trip_length_analysis.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/positive_rtts_by_trip_length.csv')
    
else:
    print("No positive RTTs data available for trip length visualizations")

print(f"\nAll results saved to Travel_Time directory:")
print(f"- weighted_tt_savings_results.csv")
print(f"- positive_rtts_summary_statistics.csv")
print(f"- rtts_distribution.png")
print(f"- uam_probability_vs_rtts.png")
print(f"- rtts_summary.png")
print(f"- simple_avg_rtts_by_trip_length.png")
print(f"- weighted_avg_rtts_by_trip_length.png")
print(f"- travel_time_by_trip_length.png")
print(f"- trip_count_by_length.png")
print(f"- weighted_savings_by_trip_length.png")
print(f"- positive_rtts_by_trip_length.csv")
##################################################################################################
# Additional Analysis: Accessibility Analysis for Positive RTTs
print("\n" + "=" * 80)
print("ACCESSIBILITY ANALYSIS FOR TRIPS WHERE UAM IS FASTER")
print("=" * 80)

if len(positive_rtts_data) > 0:
    # Catchment coverage
    both_in_catchment = ((positive_rtts_data['origin_in_catchment'] == True) & 
                         (positive_rtts_data['dest_in_catchment'] == True)).sum()
    
    print(f"\nCatchment Area Coverage:")
    print(f"- Both origin & destination in catchment: {both_in_catchment:,} trips ({both_in_catchment/len(positive_rtts_data)*100:.2f}%)")
    
    # Access/Egress distances
    avg_first_mile = positive_rtts_data['uam_first_mile'].mean()
    avg_last_mile = positive_rtts_data['uam_last_mile'].mean()
    avg_total_access = positive_rtts_data['origin_to_vertiport_dist'].mean() / 1000  # Convert to km
    avg_total_egress = positive_rtts_data['dest_to_vertiport_dist'].mean() / 1000
    
    print(f"\nAccess/Egress Distance:")
    print(f"- Average first mile distance: {avg_total_access:.2f} km")
    print(f"- Average last mile distance: {avg_total_egress:.2f} km")
    print(f"- Average first mile time: {avg_first_mile:.2f} min")
    print(f"- Average last mile time: {avg_last_mile:.2f} min")
    
    # Access modes
    print(f"\nAccess Mode Distribution:")
    origin_modes = positive_rtts_data['origin_access_mode'].value_counts()
    for mode, count in origin_modes.items():
        print(f"- {mode}: {count:,} trips ({count/len(positive_rtts_data)*100:.2f}%)")
    
    # Accessibility visualizations
    
    # Visualization 1: Access/Egress Distance Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(positive_rtts_data['origin_to_vertiport_dist']/1000, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(avg_total_access, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_total_access:.2f} km')
    ax1.set_xlabel('Distance to Origin Vertiport (km)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('First Mile Distance Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(positive_rtts_data['dest_to_vertiport_dist']/1000, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(avg_total_egress, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_total_egress:.2f} km')
    ax2.set_xlabel('Distance from Destination Vertiport (km)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Last Mile Distance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/access_egress_distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 2: Access/Egress Time Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(positive_rtts_data['uam_first_mile'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(avg_first_mile, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_first_mile:.2f} min')
    ax1.set_xlabel('First Mile Time (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('First Mile Time Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(positive_rtts_data['uam_last_mile'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(avg_last_mile, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_last_mile:.2f} min')
    ax2.set_xlabel('Last Mile Time (minutes)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Last Mile Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/access_egress_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 3: Access Mode Distribution
    plt.figure(figsize=(12, 8))
    origin_modes = positive_rtts_data['origin_access_mode'].value_counts()
    bars = plt.bar(range(len(origin_modes)), origin_modes.values, alpha=0.7, color='steelblue')
    plt.xticks(range(len(origin_modes)), origin_modes.index, rotation=45, ha='right')
    plt.xlabel('Access Mode')
    plt.ylabel('Number of Trips')
    plt.title('Access Mode Distribution (to Origin Vertiport)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mode, value) in enumerate(origin_modes.items()):
        plt.text(i, value + max(origin_modes.values) * 0.01, 
                f'{int(value):,}\n({value/len(positive_rtts_data)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/access_mode_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 4: RTTs vs Access Distance
    plt.figure(figsize=(12, 8))
    total_access_dist = (positive_rtts_data['origin_to_vertiport_dist'] + positive_rtts_data['dest_to_vertiport_dist']) / 1000
    plt.scatter(total_access_dist, positive_rtts_data['rtts'], alpha=0.6, s=2, color='purple')
    
    # Add correlation
    correlation = total_access_dist.corr(positive_rtts_data['rtts'])
    stats_text = f'Correlation: {correlation:.3f}\nAvg Total Access: {total_access_dist.mean():.2f} km'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Total Access Distance (First Mile + Last Mile) in km')
    plt.ylabel('RTTs')
    plt.title('RTTs vs Total Access Distance')
    plt.grid(True, alpha=0.3)
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/rtts_vs_access_distance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 5: Accessibility by Trip Length
    accessibility_by_length = positive_rtts_data.groupby('trip_length_bin').agg({
        'origin_to_vertiport_dist': lambda x: (x / 1000).mean(),
        'dest_to_vertiport_dist': lambda x: (x / 1000).mean(),
        'uam_first_mile': 'mean',
        'uam_last_mile': 'mean'
    }).round(2)
    
    # Filter non-empty bins
    accessibility_by_length = accessibility_by_length[accessibility_by_length.index.isin(filtered_analysis.index)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distance by trip length
    x_pos = range(len(accessibility_by_length))
    width = 0.4
    ax1.bar([x - width/2 for x in x_pos], accessibility_by_length['origin_to_vertiport_dist'], 
            width=width, label='First Mile', color='blue', alpha=0.7)
    ax1.bar([x + width/2 for x in x_pos], accessibility_by_length['dest_to_vertiport_dist'], 
            width=width, label='Last Mile', color='green', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(accessibility_by_length.index, rotation=45)
    ax1.set_xlabel('Trip Length (km)')
    ax1.set_ylabel('Average Distance (km)')
    ax1.set_title('Average Access Distance by Trip Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time by trip length
    ax2.bar([x - width/2 for x in x_pos], accessibility_by_length['uam_first_mile'], 
            width=width, label='First Mile', color='blue', alpha=0.7)
    ax2.bar([x + width/2 for x in x_pos], accessibility_by_length['uam_last_mile'], 
            width=width, label='Last Mile', color='green', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(accessibility_by_length.index, rotation=45)
    ax2.set_xlabel('Trip Length (km)')
    ax2.set_ylabel('Average Time (minutes)')
    ax2.set_title('Average Access Time by Trip Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/accessibility_by_trip_length.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Spatial Accessibility Maps
    print(f"\nGenerating spatial accessibility maps...")
    
    # Visualization 6: Spatial Distribution of Origins (where UAM is faster)
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(positive_rtts_data['originX'], positive_rtts_data['originY'], 
                         c=positive_rtts_data['rtts'], cmap='RdYlGn', 
                         alpha=0.6, s=10, vmin=0, vmax=0.3)
    cbar = plt.colorbar(scatter, label='RTTs (Travel Time Savings)')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Spatial Distribution of Trip Origins by RTTs (UAM is Faster)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_origins_by_rtts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 7: Spatial Distribution of Destinations (where UAM is faster)
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(positive_rtts_data['destinationX'], positive_rtts_data['destinationY'], 
                         c=positive_rtts_data['rtts'], cmap='RdYlGn', 
                         alpha=0.6, s=10, vmin=0, vmax=0.3)
    cbar = plt.colorbar(scatter, label='RTTs (Travel Time Savings)')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Spatial Distribution of Trip Destinations by RTTs (UAM is Faster)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_destinations_by_rtts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 8: Spatial Distribution by Access Distance
    plt.figure(figsize=(14, 10))
    total_access = (positive_rtts_data['origin_to_vertiport_dist'] + positive_rtts_data['dest_to_vertiport_dist']) / 1000
    scatter = plt.scatter(positive_rtts_data['originX'], positive_rtts_data['originY'], 
                         c=total_access, cmap='YlOrRd', 
                         alpha=0.6, s=10)
    cbar = plt.colorbar(scatter, label='Total Access Distance (km)')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Spatial Distribution of Trip Origins by Total Access Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_origins_by_access_distance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 9: Origin-Destination Flow Map (sample for clarity)
    plt.figure(figsize=(14, 10))
    # Sample 1000 trips for clarity
    sample_data = positive_rtts_data.sample(n=min(1000, len(positive_rtts_data)), random_state=42)
    
    for _, row in sample_data.iterrows():
        plt.plot([row['originX'], row['destinationX']], 
                [row['originY'], row['destinationY']], 
                'b-', alpha=0.1, linewidth=0.5)
    
    plt.scatter(sample_data['originX'], sample_data['originY'], 
               c='green', s=5, alpha=0.5, label='Origins')
    plt.scatter(sample_data['destinationX'], sample_data['destinationY'], 
               c='red', s=5, alpha=0.5, label='Destinations')
    
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Origin-Destination Flow Map (Sample of 1000 trips where UAM is Faster)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_od_flow_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 10: Hexbin Density Map for Origins
    plt.figure(figsize=(14, 10))
    hexbin = plt.hexbin(positive_rtts_data['originX'], positive_rtts_data['originY'], 
                       gridsize=50, cmap='YlOrRd', mincnt=1)
    cbar = plt.colorbar(hexbin, label='Number of Trips')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Trip Origin Density (where UAM is Faster)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_origin_density.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 11: Hexbin Density Map for Destinations
    plt.figure(figsize=(14, 10))
    hexbin = plt.hexbin(positive_rtts_data['destinationX'], positive_rtts_data['destinationY'], 
                       gridsize=50, cmap='YlOrRd', mincnt=1)
    cbar = plt.colorbar(hexbin, label='Number of Trips')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Trip Destination Density (where UAM is Faster)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_destination_density.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 12: UAM Probability by Spatial Location
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(positive_rtts_data['originX'], positive_rtts_data['originY'], 
                         c=positive_rtts_data['prob_mode_Autonomous Flying Taxi'], 
                         cmap='viridis', alpha=0.6, s=10)
    cbar = plt.colorbar(scatter, label='UAM Probability')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('UAM Probability by Trip Origin Location')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_uam_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save spatial data to CSV for GIS analysis
    spatial_data = positive_rtts_data[['originX', 'originY', 'destinationX', 'destinationY', 
                                       'rtts', 'prob_mode_Autonomous Flying Taxi', 
                                       'origin_to_vertiport_dist', 'dest_to_vertiport_dist',
                                       'trip_length', 'travel_time_Uam', 'autos_TT']].copy()
    spatial_data.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/spatial_accessibility_data.csv', index=False)
    
    print(f"\nAccessibility visualizations saved:")
    print(f"- access_egress_distance_distribution.png")
    print(f"- access_egress_time_distribution.png")
    print(f"- access_mode_distribution.png")
    print(f"- rtts_vs_access_distance.png")
    print(f"- accessibility_by_trip_length.png")
    print(f"\nSpatial accessibility maps saved:")
    print(f"- spatial_origins_by_rtts.png")
    print(f"- spatial_destinations_by_rtts.png")
    print(f"- spatial_origins_by_access_distance.png")
    print(f"- spatial_od_flow_map.png")
    print(f"- spatial_origin_density.png")
    print(f"- spatial_destination_density.png")
    print(f"- spatial_uam_probability.png")
    print(f"- spatial_accessibility_data.csv (for GIS software)")
    
else:
    print("No positive RTTs data for accessibility analysis")
#####################################################################################################
print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("• Positive RTTs = UAM is FASTER than ground transport")
print("• Simple Average = Average RTTs across all positive RTTs trips")
print("• Weighted Average = Average RTTs weighted by UAM probability")
print("• Analysis focuses only on trips where UAM provides time savings")
print("• Trip length analysis shows which distances UAM performs best")
print("=" * 80)
