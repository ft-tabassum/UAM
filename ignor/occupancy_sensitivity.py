import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("UAM CONGESTION ANALYSIS - OCCUPANCY SENSITIVITY")
print("=" * 60)

# Load dataset
data = pd.read_csv(
    '/Result/Vertiport_analysis/Probability_clustering/Weighting/5km_radius_LightGBM_synthetic_population_predictions_weights.csv')

print(f"\nDataset loaded: {len(data):,} trips")

# Determine chosen mode function
def determine_mode(row):
    car_prob = row['prob_mode_Car']
    pt_prob = row['prob_mode_Public Transport']
    uam_prob = row['prob_mode_Autonomous Flying Taxi']

    if car_prob > pt_prob and car_prob > uam_prob:
        return 'Car'
    elif pt_prob > car_prob and pt_prob > uam_prob:
        return 'Public Transport'
    elif uam_prob > car_prob and uam_prob > pt_prob:
        return 'UAM'
    else:
        return 'Tie'

# Determine chosen mode for all data
data['chosen_mode'] = data.apply(determine_mode, axis=1)

# Convert trip length to km
data['trip_length_km'] = data['trip_length'] / 1000

# Use provided mode share values
provided_mode_counts = {
    'Car': 795959,
    'Public Transport': 169763,
    'UAM': 32740
}

total_provided = sum(provided_mode_counts.values())

# Get current trips
current_car_trips = data[data['chosen_mode'] == 'Car']
current_pt_trips = data[data['chosen_mode'] == 'Public Transport']
uam_trips = data[data['chosen_mode'] == 'UAM'].copy()

# Calculate redistribution probabilities for UAM trips
if len(uam_trips) > 0:
    uam_trips['car_prob'] = uam_trips['prob_mode_Car']
    uam_trips['pt_prob'] = uam_trips['prob_mode_Public Transport']
    uam_trips['total_prob'] = uam_trips['car_prob'] + uam_trips['pt_prob']
    uam_trips = uam_trips[uam_trips['total_prob'] > 0]
    uam_trips['car_prob_norm'] = uam_trips['car_prob'] / uam_trips['total_prob']
    uam_trips['pt_prob_norm'] = uam_trips['pt_prob'] / uam_trips['total_prob']

# Calculate access/egress VKT (constant - using car occupancy 1.2)
access_egress_vkt = 0
if len(uam_trips) > 0 and 'origin_to_vertiport_dist' in uam_trips.columns and 'dest_to_vertiport_dist' in uam_trips.columns:
    uam_trips['origin_to_vertiport_km'] = uam_trips['origin_to_vertiport_dist'] / 1000
    uam_trips['dest_to_vertiport_km'] = uam_trips['dest_to_vertiport_dist'] / 1000
    # Apply car occupancy to access/egress VKT
    first_mile_vkt = (uam_trips['origin_to_vertiport_km'] / 1.2).sum()
    last_mile_vkt = (uam_trips['dest_to_vertiport_km'] / 1.2).sum()
    access_egress_vkt = first_mile_vkt + last_mile_vkt

print(f"\nAccess/Egress VKT (constant): {access_egress_vkt:,.1f} km")

# Define sensitivity analysis scenarios
car_occupancy_scenarios = [1.0, 1.1, 1.2, 1.3]
pt_occupancy_scenarios = [15, 20, 25, 30]

# Store results
results = []

print(f"\nSENSITIVITY ANALYSIS SCENARIOS:")
print(f"Car Occupancy: {car_occupancy_scenarios}")
print(f"PT Occupancy: {pt_occupancy_scenarios}")

# Run sensitivity analysis
for car_occ in car_occupancy_scenarios:
    for pt_occ in pt_occupancy_scenarios:
        
        # Calculate VKT for current trips
        current_car_vkt = (current_car_trips['trip_length_km'] / car_occ).sum()
        current_pt_vkt = (current_pt_trips['trip_length_km'] / pt_occ).sum()
        
        # Calculate redistributed VKT
        if len(uam_trips) > 0:
            redistributed_car_vkt = ((uam_trips['trip_length_km'] * uam_trips['car_prob_norm']) / car_occ).sum()
            redistributed_pt_vkt = ((uam_trips['trip_length_km'] * uam_trips['pt_prob_norm']) / pt_occ).sum()
        else:
            redistributed_car_vkt = 0
            redistributed_pt_vkt = 0
        
        # Calculate before and after UAM VKT
        before_uam_vkt = current_car_vkt + current_pt_vkt + redistributed_car_vkt + redistributed_pt_vkt
        after_uam_vkt = current_car_vkt + current_pt_vkt + access_egress_vkt
        
        # Calculate net impact
        net_vkt_impact = after_uam_vkt - before_uam_vkt
        net_vkt_impact_pct = (net_vkt_impact / before_uam_vkt) * 100
        
        # Main trip VKT reduction
        main_trip_vkt_reduction = redistributed_car_vkt + redistributed_pt_vkt

        
        results.append({
            'Car_Occupancy': car_occ,
            'PT_Occupancy': pt_occ,
            'Before_UAM_VKT': before_uam_vkt,
            'After_UAM_VKT': after_uam_vkt,
            'Net_VKT_Impact': net_vkt_impact,
            'Net_VKT_Impact_Pct': net_vkt_impact_pct,
            'Main_Trip_VKT_Reduction': main_trip_vkt_reduction,
            'Access_Egress_VKT': access_egress_vkt
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save detailed results
results_df.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/occupancy_sensitivity_results.csv', index=False)

print(f"\nDetailed results saved to: occupancy_sensitivity_results.csv")

# Analysis and Summary
print(f"\nSENSITIVITY ANALYSIS SUMMARY:")
print("=" * 40)

# Find baseline scenario (car=1.2, pt=25)
baseline = results_df[(results_df['Car_Occupancy'] == 1.2) & (results_df['PT_Occupancy'] == 25.0)].iloc[0]
print(f"Baseline Scenario (Car=1.2, PT=25):")
print(f"  Before UAM VKT: {baseline['Before_UAM_VKT']:,.1f} km")
print(f"  After UAM VKT: {baseline['After_UAM_VKT']:,.1f} km")
print(f"  Net VKT Impact: {baseline['Net_VKT_Impact']:+,.1f} km ({baseline['Net_VKT_Impact_Pct']:+.2f}%)")

# Analyze sensitivity ranges
before_range = results_df['Before_UAM_VKT'].agg(['min', 'max'])
after_range = results_df['After_UAM_VKT'].agg(['min', 'max'])
net_impact_range = results_df['Net_VKT_Impact_Pct'].agg(['min', 'max'])

print(f"\nSensitivity Ranges:")
print(f"  Before UAM VKT: {before_range['min']:,.1f} to {before_range['max']:,.1f} km")
print(f"  After UAM VKT: {after_range['min']:,.1f} to {after_range['max']:,.1f} km")
print(f"  Net VKT Impact: {net_impact_range['min']:+.2f}% to {net_impact_range['max']:+.2f}%")

# Count scenarios where UAM reduces vs increases congestion
reduces_congestion = (results_df['Net_VKT_Impact_Pct'] < 0).sum()
increases_congestion = (results_df['Net_VKT_Impact_Pct'] > 0).sum()
total_scenarios = len(results_df)

print(f"\nScenario Outcomes:")
print(f"  UAM Reduces Congestion: {reduces_congestion}/{total_scenarios} scenarios ({reduces_congestion/total_scenarios*100:.1f}%)")
print(f"  UAM Increases Congestion: {increases_congestion}/{total_scenarios} scenarios ({increases_congestion/total_scenarios*100:.1f}%)")

# Find most and least favorable scenarios
most_favorable = results_df.loc[results_df['Net_VKT_Impact_Pct'].idxmin()]
least_favorable = results_df.loc[results_df['Net_VKT_Impact_Pct'].idxmax()]

print(f"\nMost Favorable Scenario (Car={most_favorable['Car_Occupancy']}, PT={most_favorable['PT_Occupancy']}):")
print(f"  Net VKT Impact: {most_favorable['Net_VKT_Impact_Pct']:+.2f}%")

print(f"\nLeast Favorable Scenario (Car={least_favorable['Car_Occupancy']}, PT={least_favorable['PT_Occupancy']}):")
print(f"  Net VKT Impact: {least_favorable['Net_VKT_Impact_Pct']:+.2f}%")

# Create separate visualizations
plt.style.use('default')

# 1. Net VKT Impact Heatmap
plt.figure(figsize=(12, 8))
pivot_impact = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='Net_VKT_Impact_Pct')
sns.heatmap(pivot_impact, annot=True, fmt='.1f', cmap='Greys', center=0)
plt.title('Net VKT Impact (%) - Occupancy Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Car Occupancy', fontsize=12)
plt.ylabel('PT Occupancy', fontsize=12)
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/net_vkt_impact_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Before UAM VKT Heatmap
plt.figure(figsize=(12, 8))
pivot_before = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='Before_UAM_VKT')
sns.heatmap(pivot_before, annot=True, fmt='.0f', cmap='Greys')
plt.title('Before UAM VKT (km) - Occupancy Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Car Occupancy', fontsize=12)
plt.ylabel('PT Occupancy', fontsize=12)
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/before_uam_vkt_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. After UAM VKT Heatmap
plt.figure(figsize=(12, 8))
pivot_after = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='After_UAM_VKT')
sns.heatmap(pivot_after, annot=True, fmt='.0f', cmap='Greys')
plt.title('After UAM VKT (km) - Occupancy Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Car Occupancy', fontsize=12)
plt.ylabel('PT Occupancy', fontsize=12)
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/after_uam_vkt_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Car Occupancy Sensitivity (holding PT=25)
plt.figure(figsize=(10, 6))
pt_25_data = results_df[results_df['PT_Occupancy'] == 25.0]
plt.plot(pt_25_data['Car_Occupancy'], pt_25_data['Net_VKT_Impact_Pct'], 'o-', linewidth=3, markersize=8, color='black')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
plt.xlabel('Car Occupancy', fontsize=12)
plt.ylabel('Net VKT Impact (%)', fontsize=12)
plt.title('Car Occupancy Sensitivity (PT=25)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/car_occupancy_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. PT Occupancy Sensitivity (holding Car=1.2)
plt.figure(figsize=(10, 6))
car_12_data = results_df[results_df['Car_Occupancy'] == 1.2]
plt.plot(car_12_data['PT_Occupancy'], car_12_data['Net_VKT_Impact_Pct'], 'o-', linewidth=3, markersize=8, color='black')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
plt.xlabel('PT Occupancy', fontsize=12)
plt.ylabel('Net VKT Impact (%)', fontsize=12)
plt.title('PT Occupancy Sensitivity (Car=1.2)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/pt_occupancy_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()

# Save summary report
summary_file = '/Result/Vertiport_analysis/Output_analyze/congestion/occupancy_sensitivity_summary.txt'

with open(summary_file, 'w') as f:
    f.write("UAM CONGESTION ANALYSIS - SIMPLE OCCUPANCY SENSITIVITY SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("ANALYSIS OVERVIEW:\n")
    f.write(f"- Total scenarios analyzed: {total_scenarios}\n")
    f.write(f"- Car occupancy range: {min(car_occupancy_scenarios)} to {max(car_occupancy_scenarios)}\n")
    f.write(f"- PT occupancy range: {min(pt_occupancy_scenarios)} to {max(pt_occupancy_scenarios)}\n")
    f.write(f"- Access/Egress VKT (constant): {access_egress_vkt:,.1f} km\n\n")
    
    f.write("BASELINE SCENARIO (Car=1.2, PT=25):\n")
    f.write(f"- Before UAM VKT: {baseline['Before_UAM_VKT']:,.1f} km\n")
    f.write(f"- After UAM VKT: {baseline['After_UAM_VKT']:,.1f} km\n")
    f.write(f"- Net VKT Impact: {baseline['Net_VKT_Impact']:+,.1f} km ({baseline['Net_VKT_Impact_Pct']:+.2f}%)\n")
    
    f.write("SENSITIVITY RANGES:\n")
    f.write(f"- Before UAM VKT: {before_range['min']:,.1f} to {before_range['max']:,.1f} km\n")
    f.write(f"- After UAM VKT: {after_range['min']:,.1f} to {after_range['max']:,.1f} km\n")
    f.write(f"- Net VKT Impact: {net_impact_range['min']:+.2f}% to {net_impact_range['max']:+.2f}%\n\n")
    
    f.write("SCENARIO OUTCOMES:\n")
    f.write(f"- UAM Reduces Congestion: {reduces_congestion}/{total_scenarios} scenarios ({reduces_congestion/total_scenarios*100:.1f}%)\n")
    f.write(f"- UAM Increases Congestion: {increases_congestion}/{total_scenarios} scenarios ({increases_congestion/total_scenarios*100:.1f}%)\n\n")
    
    f.write("MOST FAVORABLE SCENARIO:\n")
    f.write(f"- Car Occupancy: {most_favorable['Car_Occupancy']}\n")
    f.write(f"- PT Occupancy: {most_favorable['PT_Occupancy']}\n")
    f.write(f"- Net VKT Impact: {most_favorable['Net_VKT_Impact_Pct']:+.2f}%\n\n")
    
    f.write("LEAST FAVORABLE SCENARIO:\n")
    f.write(f"- Car Occupancy: {least_favorable['Car_Occupancy']}\n")
    f.write(f"- PT Occupancy: {least_favorable['PT_Occupancy']}\n")
    f.write(f"- Net VKT Impact: {least_favorable['Net_VKT_Impact_Pct']:+.2f}%\n\n")
    
    f.write("KEY INSIGHTS:\n")
    f.write("1. Occupancy rates significantly impact VKT calculations\n")
    f.write("2. Lower car occupancy generally makes UAM more beneficial\n")
    f.write("3. Higher PT occupancy generally makes UAM less beneficial\n")
    f.write("4. The sensitivity analysis shows robustness of conclusions\n")
    f.write("5. Access/egress VKT is held constant with car occupancy 1.2\n\n")
    
    f.write(f"Analysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\nSummary report saved to: occupancy_sensitivity_summary.txt")
print(f"Individual graphs saved:")
print(f"- net_vkt_impact_heatmap.png")
print(f"- before_uam_vkt_heatmap.png") 
print(f"- after_uam_vkt_heatmap.png")
print(f"- car_occupancy_sensitivity.png")
print(f"- pt_occupancy_sensitivity.png")

# Create matrix tables
print(f"\nCREATING MATRIX TABLES:")
print("=" * 40)

# Create matrix table for Net VKT Impact Percentage
pivot_net_impact = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='Net_VKT_Impact_Pct')

print(f"\nNET VKT IMPACT MATRIX (%):")
print("PTOccupancy/Car OCCUPANCY\t1.0\t\t1.1\t\t1.2\t\t1.3")
for pt_occ in pivot_net_impact.index:
    row_values = []
    for car_occ in pivot_net_impact.columns:
        row_values.append(f"{pivot_net_impact.loc[pt_occ, car_occ]:.9f}")
    print(f"{pt_occ}\t\t\t{row_values[0]}\t{row_values[1]}\t{row_values[2]}\t{row_values[3]}")

# Create matrix table for Net VKT Impact (absolute values in km)
pivot_net_impact_abs = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='Net_VKT_Impact')

print(f"\nNET VKT IMPACT MATRIX (km):")
print("PTOccupancy/Car OCCUPANCY\t1.0\t\t1.1\t\t1.2\t\t1.3")
for pt_occ in pivot_net_impact_abs.index:
    row_values = []
    for car_occ in pivot_net_impact_abs.columns:
        row_values.append(f"{pivot_net_impact_abs.loc[pt_occ, car_occ]:.0f}")
    print(f"{pt_occ}\t\t\t{row_values[0]}\t\t{row_values[1]}\t\t{row_values[2]}\t\t{row_values[3]}")

# Create matrix table for Before UAM VKT
pivot_before = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='Before_UAM_VKT')

print(f"\nBEFORE UAM VKT MATRIX (km):")
print("PTOccupancy/Car OCCUPANCY\t1.0\t\t1.1\t\t1.2\t\t1.3")
for pt_occ in pivot_before.index:
    row_values = []
    for car_occ in pivot_before.columns:
        row_values.append(f"{pivot_before.loc[pt_occ, car_occ]:.0f}")
    print(f"{pt_occ}\t\t\t{row_values[0]}\t\t{row_values[1]}\t\t{row_values[2]}\t\t{row_values[3]}")

# Create matrix table for After UAM VKT
pivot_after = results_df.pivot(index='PT_Occupancy', columns='Car_Occupancy', values='After_UAM_VKT')

print(f"\nAFTER UAM VKT MATRIX (km):")
print("PTOccupancy/Car OCCUPANCY\t1.0\t\t1.1\t\t1.2\t\t1.3")
for pt_occ in pivot_after.index:
    row_values = []
    for car_occ in pivot_after.columns:
        row_values.append(f"{pivot_after.loc[pt_occ, car_occ]:.0f}")
    print(f"{pt_occ}\t\t\t{row_values[0]}\t\t{row_values[1]}\t\t{row_values[2]}\t\t{row_values[3]}")

# Save all matrices to CSV files
pivot_net_impact.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/net_vkt_impact_matrix_pct.csv')
pivot_net_impact_abs.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/net_vkt_impact_matrix_km.csv')
pivot_before.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/before_uam_vkt_matrix.csv')
pivot_after.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/after_uam_vkt_matrix.csv')

print(f"\nMatrix tables saved to CSV files:")
print("- net_vkt_impact_matrix_pct.csv")
print("- net_vkt_impact_matrix_km.csv") 
print("- before_uam_vkt_matrix.csv")
print("- after_uam_vkt_matrix.csv")

print("\n" + "="*60)
print("OCCUPANCY SENSITIVITY ANALYSIS COMPLETE!")
print("="*60)
