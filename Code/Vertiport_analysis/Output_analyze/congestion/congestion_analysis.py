import pandas as pd
import numpy as np

print("UAM CONGESTION ANALYSIS")
print("=" * 55)

# Define occupancy rates
CAR_OCCUPANCY = 1.2  # average passenger per car
PT_OCCUPANCY = 25.0  # people per PT vehicle (assumption)

print(f"\nOCCUPANCY RATES:")
print(f"   Car: {CAR_OCCUPANCY} people/vehicle")
print(f"   Public Transport: {PT_OCCUPANCY} people/vehicle")

# Load dataset
data = pd.read_csv(
    'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

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

# Use provided mode share values (collected from modeShare.py)
provided_mode_counts = {
    'Car': 795959,
    'Public Transport': 169763,
    'UAM': 32740 }

total_provided = sum(provided_mode_counts.values())

print("\nPROVIDED MODE SHARE VALUES (collected from modeShare.py):")
print("-" * 55)
for mode, count in provided_mode_counts.items():
    share = (count / total_provided) * 100
    print(f"{mode:20}: {count:8,} trips ({share:6.2f}%)")

print(f"\nTotal provided trips: {total_provided:,}")

# Calculate VKT for current car and PT trips using provided mode shares
current_car_trips = data[data['chosen_mode'] == 'Car']
current_pt_trips = data[data['chosen_mode'] == 'Public Transport']

# Calculate VKT: For each trip, VKT contribution = distance ÷ occupancy
current_car_vkt = (current_car_trips['trip_length_km'] / CAR_OCCUPANCY).sum()
current_pt_vkt = (current_pt_trips['trip_length_km'] / PT_OCCUPANCY).sum()

print(f"\nVKT CALCULATION:")
print("=" * 40)
print(f"Car Analysis:")
print(f"  Number of trips: {len(current_car_trips):,}")
print(f"  VKT: {current_car_vkt:,.1f} km")

print(f"\nPT Analysis:")
print(f"  Number of trips: {len(current_pt_trips):,}")
print(f"  VKT: {current_pt_vkt:,.1f} km")

# Process UAM trips
uam_trips = data[data['chosen_mode'] == 'UAM'].copy()

redistributed_car_vkt = 0
redistributed_pt_vkt = 0
access_egress_vkt = 0

if len(uam_trips) > 0:
    # Calculate redistribution probabilities
    uam_trips['car_prob'] = uam_trips['prob_mode_Car']
    uam_trips['pt_prob'] = uam_trips['prob_mode_Public Transport']
    uam_trips['total_prob'] = uam_trips['car_prob'] + uam_trips['pt_prob']
    uam_trips = uam_trips[uam_trips['total_prob'] > 0]
    uam_trips['car_prob_norm'] = uam_trips['car_prob'] / uam_trips['total_prob']
    uam_trips['pt_prob_norm'] = uam_trips['pt_prob'] / uam_trips['total_prob']

    # Calculate redistributed VKT properly: (Number of Vehicles × Distance) for each trip
    redistributed_car_vkt = ((uam_trips['trip_length_km'] * uam_trips['car_prob_norm']) / CAR_OCCUPANCY).sum()
    redistributed_pt_vkt = ((uam_trips['trip_length_km'] * uam_trips['pt_prob_norm']) / PT_OCCUPANCY).sum()

    # Calculate access/egress VKT
    if 'origin_to_vertiport_dist' in uam_trips.columns and 'dest_to_vertiport_dist' in uam_trips.columns:
        uam_trips['origin_to_vertiport_km'] = uam_trips['origin_to_vertiport_dist'] / 1000
        uam_trips['dest_to_vertiport_km'] = uam_trips['dest_to_vertiport_dist'] / 1000

        first_mile_vkt = uam_trips['origin_to_vertiport_km'].sum()
        last_mile_vkt = uam_trips['dest_to_vertiport_km'].sum()
        access_egress_vkt = first_mile_vkt + last_mile_vkt

# Use provided values for analysis
uam_trips_total = provided_mode_counts['UAM']
print(f"\nUsing provided UAM trips: {uam_trips_total:,}")

if uam_trips_total > 0:
    print("\nVKT CALCULATION USING PROVIDED MODE SHARES:")
    print("-" * 50)

    # Use VKT calculations directly (no scaling needed since we use provided mode shares)

    print(f"Using VKT calculations from data sample:")
    print(f"   Note: These represent the congestion impact based on provided mode shares")

    # BEFORE UAM: All trips use ground transport
    print("\nBEFORE UAM (All trips use ground transport):")
    before_uam_vkt = current_car_vkt + current_pt_vkt + redistributed_car_vkt + redistributed_pt_vkt

    print(f"   Current Car VKT: {current_car_vkt:,.1f} km")
    print(f"   Current PT VKT: {current_pt_vkt:,.1f} km")
    print(f"   Redistributed Car VKT: {redistributed_car_vkt:,.1f} km")
    print(f"   Redistributed PT VKT: {redistributed_pt_vkt:,.1f} km")
    print(f"   TOTAL BEFORE UAM: {before_uam_vkt:,.1f} km")

    # AFTER UAM: Current ground trips + access/egress trips
    print("\nAFTER UAM (Current ground trips + access/egress):")
    after_uam_vkt = current_car_vkt + current_pt_vkt + access_egress_vkt

    print(f"   Current Car VKT: {current_car_vkt:,.1f} km")
    print(f"   Current PT VKT: {current_pt_vkt:,.1f} km")
    print(f"   Access/Egress VKT: {access_egress_vkt:,.1f} km (no occupancy - vehicle trips)")
    print(f"   TOTAL AFTER UAM: {after_uam_vkt:,.1f} km")

    # Calculate net impact
    net_vkt_impact = after_uam_vkt - before_uam_vkt
    net_vkt_impact_pct = (net_vkt_impact / before_uam_vkt) * 100

    print(f"\nNET VKT IMPACT:")
    print(f"   Before UAM: {before_uam_vkt:,.1f} km")
    print(f"   After UAM: {after_uam_vkt:,.1f} km")
    print(f"   Difference: {net_vkt_impact:+,.1f} km")
    print(f"   Percentage: {net_vkt_impact_pct:+.2f}%")

    # Main trip VKT reduction
    main_trip_vkt_reduction = redistributed_car_vkt + redistributed_pt_vkt

    # Conclusion
    print(f"\nCONCLUSION:")
    if net_vkt_impact > 0:
        print(f"   UAM INCREASES congestion by {net_vkt_impact_pct:.2f}%")
        print(
            f"   Access/egress VKT ({access_egress_vkt:,.1f} km) > Main trip VKT reduction ({main_trip_vkt_reduction:,.1f} km)")
    else:
        print(f"   UAM REDUCES congestion by {abs(net_vkt_impact_pct):.2f}%")
        print(
            f"   Main trip VKT reduction ({main_trip_vkt_reduction:,.1f} km) > Access/egress VKT ({access_egress_vkt:,.1f} km)")

    # Break-even analysis
    break_even_ratio = main_trip_vkt_reduction / access_egress_vkt
    print(f"\nBREAK-EVEN ANALYSIS:")
    print(f"   Main trip VKT reduction: {main_trip_vkt_reduction:,.1f} km")
    print(f"   Access/egress VKT: {access_egress_vkt:,.1f} km")
    print(f"   Break-even ratio: {break_even_ratio:.2f}")

    if break_even_ratio > 1:
        print(f"   UAM reduces congestion (ratio > 1)")
    else:
        print(f"   UAM increases congestion (ratio < 1)")

    # Save results
    results = {
        'Metric': [
            'Total Trips (Provided)',
            'UAM Trips (Provided)',
            'UAM Mode Share (%) (Provided)',
            'Car Occupancy',
            'PT Occupancy',
            'Before UAM VKT (km)',
            'After UAM VKT (km)',
            'Net VKT Impact (km)',
            'Net VKT Impact (%)',
            'Main Trip VKT Reduction (km)',
            'Access/Egress VKT (km)',
            'Break-even Ratio'
        ],
        'Value': [
            total_provided,
            uam_trips_total,
            (uam_trips_total / total_provided) * 100,
            CAR_OCCUPANCY,
            PT_OCCUPANCY,
            before_uam_vkt,
            after_uam_vkt,
            net_vkt_impact,
            net_vkt_impact_pct,
            main_trip_vkt_reduction,
            access_egress_vkt,
            break_even_ratio
        ]
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/congestion_analysis.csv',
        index=False)

    print(f"\nFixed mode share results saved to: congestion_analysiscsv")

    # Save comprehensive text summary
    summary_file = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/congestion/congestion_analysis_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("UAM CONGESTION ANALYSIS - COMPREHENSIVE SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("OCCUPANCY RATES:\n")
        f.write(f"   Car: {CAR_OCCUPANCY} people/vehicle\n")
        f.write(f"   Public Transport: {PT_OCCUPANCY} people/vehicle\n\n")

        f.write("PROVIDED MODE SHARE VALUES (collected from modeShare.py):\n")
        f.write("-" * 55 + "\n")
        for mode, count in provided_mode_counts.items():
            share = (count / total_provided) * 100
            f.write(f"{mode:20}: {count:8,} trips ({share:6.2f}%)\n")
        f.write(f"\nTotal provided trips: {total_provided:,}\n\n")

        f.write("VKT CALCULATION:\n")
        f.write("=" * 40 + "\n")
        f.write(f"Car Analysis:\n")
        f.write(f"  Number of trips: {len(current_car_trips):,}\n")
        f.write(f"  VKT: {current_car_vkt:,.1f} km\n\n")
        
        f.write(f"PT Analysis:\n")
        f.write(f"  Number of trips: {len(current_pt_trips):,}\n")
        f.write(f"  VKT: {current_pt_vkt:,.1f} km\n\n")

        f.write("FINAL VKT CALCULATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Before UAM VKT: {before_uam_vkt:,.1f} km\n")
        f.write(f"After UAM VKT: {after_uam_vkt:,.1f} km\n")
        f.write(f"Net VKT Impact: {net_vkt_impact:+,.1f} km ({net_vkt_impact_pct:+.2f}%)\n")
        f.write(f"Main Trip VKT Reduction: {main_trip_vkt_reduction:,.1f} km\n")
        f.write(f"Access/Egress VKT: {access_egress_vkt:,.1f} km\n")
        f.write(f"Break-even Ratio: {break_even_ratio:.2f}\n\n")

        f.write("CONCLUSION:\n")
        f.write("-" * 20 + "\n")
        if net_vkt_impact > 0:
            f.write(f"UAM INCREASES congestion by {net_vkt_impact_pct:.2f}%\n")
            f.write(
                f"Access/egress VKT ({access_egress_vkt:,.1f} km) > Main trip VKT reduction ({main_trip_vkt_reduction:,.1f} km)\n")
        else:
            f.write(f"UAM REDUCES congestion by {abs(net_vkt_impact_pct):.2f}%\n")
            f.write(
                f"Main trip VKT reduction ({main_trip_vkt_reduction:,.1f} km) > Access/egress VKT ({access_egress_vkt:,.1f} km)\n")

        f.write(f"\nAnalysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nComprehensive text summary saved to: congestion_analysis_summary.txt")

print("\nFixed mode share analysis complete!")




