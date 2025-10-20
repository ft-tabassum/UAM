import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

data = pd.read_csv(
    'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/5km_radius_LightGBM_synthetic_population_predictions_weights.csv')

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
print(f"Percentage of trips where UAM is faster: {len(positive_rtts_data) / len(data) * 100:.2f}%")
print("=" * 80)

if len(positive_rtts_data) > 0:
    # Calculate Average RTTs for positive RTTs only
    # Average (unweighted) for positive RTTs
    avg_positive_rtts = positive_rtts_data['rtts'].mean()
    
    # Weighted average using UAM probability as weights for positive RTTs
    weighted_avg_positive_rtts = (positive_rtts_data['rtts'] * positive_rtts_data[
        'prob_mode_Autonomous Flying Taxi']).sum() / positive_rtts_data['prob_mode_Autonomous Flying Taxi'].sum()
    
    # Additional statistics for positive RTTs
    median_positive_rtts = positive_rtts_data['rtts'].median()
    std_positive_rtts = positive_rtts_data['rtts'].std()
    
    # Print results for positive RTTs
    print("\nPOSITIVE RTTs ANALYSIS:")
    print("-" * 60)
    print(f"Average RTTs (positive only): {avg_positive_rtts:.4f} ({avg_positive_rtts * 100:.2f}%)")
    print(
        f"Weighted Average RTTs (positive only): {weighted_avg_positive_rtts:.4f} ({weighted_avg_positive_rtts * 100:.2f}%)")
    print(f"Median RTTs (positive only): {median_positive_rtts:.4f} ({median_positive_rtts * 100:.2f}%)")
    print(f"Standard Deviation (positive only): {std_positive_rtts:.4f}")
    print(f"Trips with positive RTTs: {len(positive_rtts_data):,}")
    print("-" * 60)
else:
    print("\nWARNING: No trips found where UAM is faster than ground transport!")
    print("All RTTs values are negative or zero.")
    avg_positive_rtts = 0
    weighted_avg_positive_rtts = 0
    median_positive_rtts = 0
    std_positive_rtts = 0

# Basic RTTs visualizations removed - focusing on trip length analysis only

# Save the results to a new CSV file
data.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/weighted_tt_savings_results.csv',
            index=False)

# Save summary statistics for positive RTTs
if len(positive_rtts_data) > 0:
    summary_stats = {
        'Metric': ['Average RTTs (Positive)', 'Weighted Average RTTs (Positive)', 'Median RTTs (Positive)',
                   'Standard Deviation (Positive)', 'Total Positive RTTs Trips', 'Total Trips Analyzed'],
        'Value': [avg_positive_rtts, weighted_avg_positive_rtts, median_positive_rtts, std_positive_rtts,
                  len(positive_rtts_data), len(data)],
        'Percentage': [f"{avg_positive_rtts * 100:.2f}%", f"{weighted_avg_positive_rtts * 100:.2f}%",
                       f"{median_positive_rtts * 100:.2f}%", f"{std_positive_rtts * 100:.2f}%",
                       f"{len(positive_rtts_data):,}", f"{len(data):,}"]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(
        'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/positive_rtts_summary_statistics.csv',
        index=False)
else:
    print("No positive RTTs data to save in summary statistics")

print(f"\nInitial results saved to:")
print(f"- weighted_tt_savings_results.csv")
print(f"- positive_rtts_summary_statistics.csv")
print("\nNote: Basic RTTs, Spatial maps, and Accessibility analysis removed - Focusing on Trip Length Analysis only")

# Additional Analysis: Trip Length Analysis for Positive RTTs
print("\n" + "=" * 80)
print("TRIP LENGTH ANALYSIS FOR POSITIVE RTTs (UAM is FASTER)")
print("=" * 80)

if len(positive_rtts_data) > 0:
    # Convert trip_length from meters to kilometers for positive RTTs data
    positive_rtts_data['trip_length_km'] = positive_rtts_data['trip_length'] / 1000
    
    # Create trip length bins for positive RTTs (extended to 150km with 5km intervals from 20km onwards)
    positive_rtts_data['trip_length_bin'] = pd.cut(positive_rtts_data['trip_length_km'], 
                                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150],
                                                   labels=['0-5km', '5-10km', '10-15km', '15-20km', '20-25km', '25-30km', '30-35km', '35-40km', '40-45km', '45-50km', 
                                                           '50-55km', '55-60km', '60-65km', '65-70km', '70-75km', '75-80km', '80-85km', '85-90km', '90-95km', '95-100km',
                                                           '100-105km', '105-110km', '110-115km', '115-120km', '120-125km', '125-130km', '130-135km', '135-140km', '140-145km', '145-150km'])
    
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
            lambda x: (x['rtts'] * x['prob_mode_Autonomous Flying Taxi']).sum() / x[
                'prob_mode_Autonomous Flying Taxi'].sum()
        )
    ).round(4)
    
    # Rename columns for clarity
    positive_trip_length_analysis.columns = [
        'Total_Weighted_TT_Savings', 
        'Avg_RTTs',
        'Avg_UAM_Travel_Time', 
        'Avg_Auto_Travel_Time',
        'Trip_Count',
        'Weighted_Avg_RTTs'
    ]
    
    # Add percentage columns
    positive_trip_length_analysis['Avg_RTTs_Pct'] = (positive_trip_length_analysis['Avg_RTTs'] * 100).round(2)
    positive_trip_length_analysis['Weighted_Avg_RTTs_Pct'] = (
                positive_trip_length_analysis['Weighted_Avg_RTTs'] * 100).round(2)
    
    # Print results for positive RTTs by trip length
    print("\nPositive RTTs Trip Length Analysis:")
    print("-" * 80)
    for bin_name, row in positive_trip_length_analysis.iterrows():
        if row['Trip_Count'] > 0:  # Only show bins with data
            print(f"{bin_name:>10}: {row['Trip_Count']:>8,} trips | "
                  f"Avg RTTs: {row['Avg_RTTs_Pct']:>7.2f}% | "
                  f"Weighted RTTs: {row['Weighted_Avg_RTTs_Pct']:>7.2f}% | "
                  f"Avg UAM TT: {row['Avg_UAM_Travel_Time']:>6.1f}min | "
                  f"Avg Auto TT: {row['Avg_Auto_Travel_Time']:>6.1f}min")
else:
    print("No positive RTTs data available for trip length analysis")
    positive_trip_length_analysis = pd.DataFrame()

# Create separate visualizations for trip length analysis of positive RTTs

if len(positive_trip_length_analysis) > 0 and len(
        positive_trip_length_analysis[positive_trip_length_analysis['Trip_Count'] > 0]) > 0:
    
    # Filter out empty bins
    filtered_analysis = positive_trip_length_analysis[positive_trip_length_analysis['Trip_Count'] > 0]
    
    # Visualization 1: Combined - Avg RTTs and Weighted Avg RTTs by Trip Length
    plt.figure(figsize=(16, 10))  # Larger figure size
    x_pos = range(len(filtered_analysis))
    
    # Create custom labels showing only upper bounds: 20, 30, 40, ..., 100
    x_labels = []
    for label in filtered_analysis.index:
        # Extract upper bound from labels like '0-5km', '20-30km', '90-100km'
        if '-' in str(label):
            upper_bound = str(label).split('-')[1].replace('km', '')
            x_labels.append(upper_bound)
        else:
            x_labels.append(str(label).replace('km', ''))

    # Line plot with different colors and markers - increased sizes
    plt.plot(x_pos, filtered_analysis['Avg_RTTs_Pct'], 
             marker='o', markersize=12, linewidth=4.5, color='royalblue', 
             label='Average rtts', linestyle='-')
    plt.plot(x_pos, filtered_analysis['Weighted_Avg_RTTs_Pct'], 
             marker='s', markersize=12, linewidth=4.5, color='darkorange', 
             label='Weighted Average rtts', linestyle='--')


    # labels 
    plt.xlabel('Distance (km)', fontsize=18, fontweight='bold')
    plt.ylabel('Travel time savings ratio(%)', fontsize=18, fontweight='bold')
    plt.xticks(x_pos, x_labels, rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.ylim(bottom=0)  # Start y-axis from 0 to reduce gap
    plt.tight_layout()

    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/rtts_by_trip_length.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    
    # Visualization 2: Travel Time Comparison by Trip Length (10km intervals: 20-30, 30-40, ..., 140-150)
    # Create separate bins for travel time analysis
    positive_rtts_data_tt = positive_rtts_data.copy()
    positive_rtts_data_tt['trip_length_bin_tt'] = pd.cut(positive_rtts_data_tt['trip_length_km'],
                                                          bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
                                                          labels=['0-20km', '20-30km', '30-40km', '40-50km', '50-60km', '60-70km', 
                                                                  '70-80km', '80-90km', '90-100km', '100-110km', '110-120km', 
                                                                  '120-130km', '130-140km', '140-150km'])
    
    # Calculate analysis by trip length for travel time
    tt_analysis = positive_rtts_data_tt.groupby('trip_length_bin_tt').agg({
        'travel_time_Uam': 'mean',
        'autos_TT': 'mean',
        'trip_id': 'count'
    }).round(2)
    
    # Filter only bins with data
    tt_analysis = tt_analysis[tt_analysis['trip_id'] > 0]
    
    if len(tt_analysis) > 0:
        plt.figure(figsize=(12, 8))
        x_pos = range(len(tt_analysis))
        
        # Remove 'km' from x-axis labels
        x_labels_tt = [str(label).replace('km', '') for label in tt_analysis.index]
        
        uam_bars = plt.bar([x - 0.2 for x in x_pos], tt_analysis['travel_time_Uam'],
                            width=0.4, label='UAM', color='blue', alpha=0.7)
        auto_bars = plt.bar([x + 0.2 for x in x_pos], tt_analysis['autos_TT'],
                             width=0.4, label='Auto', color='orange', alpha=0.7)
        
        # Add value labels on bars
        for i, (uam_val, auto_val) in enumerate(zip(tt_analysis['travel_time_Uam'], tt_analysis['autos_TT'])):
            plt.text(i - 0.2, uam_val + 0.5, f'{uam_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            plt.text(i + 0.2, auto_val + 0.5, f'{auto_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # No title
        plt.xlabel('Distance (km)', fontsize=12, fontweight='bold')
        plt.ylabel('Travel time (min)', fontsize=12, fontweight='bold')
        plt.xticks(x_pos, x_labels_tt, rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/travel_time_by_trip_length.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualization 3: RTTs and Weighted RTTs with Trip Count (raw values, not percentage)
    # Calculate total RTTs and Weighted RTTs by trip length
    rtts_with_counts = positive_rtts_data.groupby('trip_length_bin').agg({
        'rtts': 'sum',  # Total RTTs (not averaged)
        'Weighted_TT_Savings': 'sum',  # This is weighted RTTs
        'trip_id': 'count'
    }).round(2)
    
    # Filter only bins with data
    rtts_with_counts = rtts_with_counts[rtts_with_counts['trip_id'] > 0]
    
    if len(rtts_with_counts) > 0:
        fig, ax1 = plt.subplots(figsize=(14, 8))
        x_pos = range(len(rtts_with_counts))
        
        # Create custom labels showing only upper bounds
        x_labels_rtts = []
        for label in rtts_with_counts.index:
            if '-' in str(label):
                upper_bound = str(label).split('-')[1].replace('km', '')
                x_labels_rtts.append(upper_bound)
            else:
                x_labels_rtts.append(str(label).replace('km', ''))
        
        # Plot RTTs and Weighted RTTs on primary y-axis
        ax1.plot(x_pos, rtts_with_counts['rtts'], marker='o', markersize=8, linewidth=2.5, 
                color='royalblue', label='RTTs (total)', linestyle='-')
        ax1.plot(x_pos, rtts_with_counts['Weighted_TT_Savings'], marker='s', markersize=8, linewidth=2.5, 
                color='darkorange', label='Weighted RTTs (total)', linestyle='--')
        ax1.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('RTTs Value (total)', fontsize=12, fontweight='bold', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
        
        # Plot trip count on secondary y-axis
        ax2 = ax1.twinx()
        ax2.bar(x_pos, rtts_with_counts['trip_id'], alpha=0.3, color='green', width=0.6, label='Number of Trips')
        ax2.set_ylabel('Number of Trips', fontsize=12, fontweight='bold', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Set x-axis labels
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels_rtts, rotation=0, ha='center')
        
        # No title - removed
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/rtts_weighted_with_trip_count.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save trip length analysis for positive RTTs
    positive_trip_length_analysis.to_csv(
        'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Travel_Time/positive_rtts_by_trip_length.csv')
    
else:
    print("No positive RTTs data available for trip length visualizations")

print(f"\nAll results saved to Travel_Time directory:")
print(f"- weighted_tt_savings_results.csv")
print(f"- positive_rtts_summary_statistics.csv")
print(f"- rtts_by_trip_length.png (Avg RTTs and Weighted Avg RTTs line chart)")
print(f"- travel_time_by_trip_length.png (10km intervals: 20-30, 30-40...140-150)")
print(f"- rtts_weighted_with_trip_count.png (RTTs and Weighted RTTs with trip count)")
print(f"- positive_rtts_by_trip_length.csv")

# Trip Length Analysis only

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("• Positive RTTs = UAM is FASTER than ground transport")
print("• Avg RTTs = Average RTTs across all positive RTTs trips")
print("• Weighted Average = Average RTTs weighted by UAM probability")
print("• Analysis focuses only on trips where UAM provides time savings")
print("• Trip length analysis shows which distances UAM performs best")
print("• Trip length bins: 5km intervals from 20km to 150km (20, 25, 30, 35...150)")
print("=" * 80)

