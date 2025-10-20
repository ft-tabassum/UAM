import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def analyze_vertiport_coverage():
    """
    Analyze how many demand points are covered by the 74 vertiports
    """
    print("Analyzing vertiport coverage...")
    
    # Create output directory if it doesn't exist
    output_dir = "D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/Coverage_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    output_file = os.path.join(output_dir, "5km_coverage_analysis_report.txt")
    
    # Initialize report content
    report_content = []
    
    def add_to_report(text):
        """Helper function to add text to both console and report"""
        print(text)
        report_content.append(text)
    
    # Load the prediction results
    try:
        # The file is very large, so we'll read it in chunks
        add_to_report("Loading prediction data...")
        df = pd.read_csv(
            "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/5km_radius_LightGBM_synthetic_population_predictions_weights.csv",
            low_memory=False)
        add_to_report(f"Loaded {len(df):,} demand points")
        
    except Exception as e:
        add_to_report(f"Error loading data: {e}")
        return
    
    # Load vertiport coordinates
    try:
        vertiports = pd.read_csv(
            "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid/5km_radius_optimized_vertiport_coords_final.csv")
        add_to_report(f"Loaded {len(vertiports)} vertiports")
    except Exception as e:
        add_to_report(f"Error loading vertiport coordinates: {e}")
        return
    
    # Analysis 1: Demand Points Overview
    add_to_report("\n" + "="*60)
    add_to_report("STEP 1: TOTAL DEMAND POINTS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("Each trip from origin to destination represents ONE demand point.")
    add_to_report("")
    
    # Step 1: Count total trips = total demand points
    total_trips = len(df)
    total_demand_points = total_trips  # Each trip = 1 demand point
    
    add_to_report(f"Total number of trips: {total_trips:,}")
    add_to_report(f"Total demand points: {total_demand_points:,} (1 trip = 1 demand point)")
    
    # Calculate coverage flags for later use
    both_covered = df['origin_in_catchment'] & df['dest_in_catchment']
    only_origin_covered = df['origin_in_catchment'] & ~df['dest_in_catchment']
    only_dest_covered = ~df['origin_in_catchment'] & df['dest_in_catchment']
    neither_covered = ~df['origin_in_catchment'] & ~df['dest_in_catchment']
    
    # Calculate individual endpoint coverage for supplementary info
    origins_in_catchment = df['origin_in_catchment'].sum()
    dests_in_catchment = df['dest_in_catchment'].sum()
    
    add_to_report(f"\nSupplementary Information:")
    add_to_report(f"  Origins within 5km of vertiports: {origins_in_catchment:,} ({origins_in_catchment/total_trips*100:.1f}%)")
    add_to_report(f"  Destinations within 5km of vertiports: {dests_in_catchment:,} ({dests_in_catchment/total_trips*100:.1f}%)")
    
    # Analysis 2: Potential UAM Trips Calculation
    add_to_report("\n" + "="*60)
    add_to_report("STEP 2: POTENTIAL UAM TRIPS CALCULATION")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("Potential UAM trips are calculated from ALL demand points based on ML-predicted")
    add_to_report("UAM probabilities, regardless of whether they can be served by the current")
    add_to_report("vertiport network. This represents total potential UAM demand.")
    add_to_report("")
    add_to_report("CALCULATION METHOD:")
    add_to_report("Potential UAM Trips = Total Trips × Average UAM Probability")
    add_to_report("")
    
    # Find UAM probability column
    uam_prob_col = 'prob_mode_Autonomous Flying Taxi'
    
    # Verify the column exists
    if uam_prob_col not in df.columns:
        add_to_report(f"Error: UAM probability column '{uam_prob_col}' not found in data")
        prob_cols = [col for col in df.columns if 'prob_mode_' in col]
        add_to_report(f"Available probability columns: {prob_cols}")
        return
    
    # Calculate average UAM probability and potential UAM trips
    avg_uam_probability = df[uam_prob_col].mean()
    total_potential_uam_trips = total_trips * avg_uam_probability
    
    add_to_report(f"Average UAM probability: {avg_uam_probability:.4f} ({avg_uam_probability*100:.2f}%)")
    add_to_report(f"Total trips: {total_trips:,}")
    add_to_report(f"Potential UAM trips: {total_trips:,} × {avg_uam_probability:.4f} = {total_potential_uam_trips:,.0f} trips")
    
    # Analysis 3: Access Mode Distribution
    add_to_report("\n" + "="*60)
    add_to_report("ACCESS MODE DISTRIBUTION")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("This analysis shows how people access vertiports - either by walking")
    add_to_report("(within 1km catchment) or by car (within 5km catchment).")
    add_to_report("Walking access is more convenient but has a smaller catchment area.")
    add_to_report("")
    
    origin_walking = (df['origin_access_mode'] == 'walk').sum()
    origin_car = (df['origin_access_mode'] == 'car').sum()
    dest_walking = (df['dest_access_mode'] == 'walk').sum()
    dest_car = (df['dest_access_mode'] == 'car').sum()
    
    add_to_report(f"ORIGIN ACCESS MODES:")
    add_to_report(f"  Walking: {origin_walking:,} ({origin_walking/total_trips*100:.1f}%)")
    add_to_report(f"  Car: {origin_car:,} ({origin_car/total_trips*100:.1f}%)")
    
    add_to_report(f"\nDESTINATION ACCESS MODES:")
    add_to_report(f"  Walking: {dest_walking:,} ({dest_walking/total_trips*100:.1f}%)")
    add_to_report(f"  Car: {dest_car:,} ({dest_car/total_trips*100:.1f}%)")
    
    # Analysis 4: Distance Statistics
    add_to_report("\n" + "="*60)
    add_to_report("DISTANCE STATISTICS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("This analysis shows the distance statistics from demand points to their")
    add_to_report("assigned vertiports. Distances are measured in meters.")
    add_to_report("")
    
    origin_dist_stats = df['origin_to_vertiport_dist'].describe()
    dest_dist_stats = df['dest_to_vertiport_dist'].describe()
    
    add_to_report(f"ORIGIN TO VERTIPORT DISTANCES:")
    add_to_report(f"  Mean: {origin_dist_stats['mean']:.1f}m ({origin_dist_stats['mean']/1000:.2f}km)")
    add_to_report(f"  Median: {origin_dist_stats['50%']:.1f}m ({origin_dist_stats['50%']/1000:.2f}km)")
    add_to_report(f"  Max: {origin_dist_stats['max']:.1f}m ({origin_dist_stats['max']/1000:.2f}km)")
    
    add_to_report(f"\nDESTINATION TO VERTIPORT DISTANCES:")
    add_to_report(f"  Mean: {dest_dist_stats['mean']:.1f}m ({dest_dist_stats['mean']/1000:.2f}km)")
    add_to_report(f"  Median: {dest_dist_stats['50%']:.1f}m ({dest_dist_stats['50%']/1000:.2f}km)")
    add_to_report(f"  Max: {dest_dist_stats['max']:.1f}m ({dest_dist_stats['max']/1000:.2f}km)")
    
    # Analysis 5: Vertiport Assignment Distribution
    add_to_report("\n" + "="*60)
    add_to_report("VERTIPORT ASSIGNMENT DISTRIBUTION")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("This analysis shows how demand points are distributed across the 74 vertiports.")
    add_to_report("It helps identify if some vertiports are overloaded or underutilized.")
    add_to_report("")
    
    # Count how many demand points are assigned to each vertiport
    origin_assignments = df['uam_origin_vertiport'].value_counts().sort_index()
    dest_assignments = df['uam_dest_vertiport'].value_counts().sort_index()
    
    add_to_report(f"ORIGIN VERTIPORT ASSIGNMENTS:")
    add_to_report(f"  Min demand points per vertiport: {origin_assignments.min():,}")
    add_to_report(f"  Max demand points per vertiport: {origin_assignments.max():,}")
    add_to_report(f"  Mean demand points per vertiport: {origin_assignments.mean():.1f}")
    add_to_report(f"  Median demand points per vertiport: {origin_assignments.median():.1f}")
    
    add_to_report(f"\nDESTINATION VERTIPORT ASSIGNMENTS:")
    add_to_report(f"  Min demand points per vertiport: {dest_assignments.min():,}")
    add_to_report(f"  Max demand points per vertiport: {dest_assignments.max():,}")
    add_to_report(f"  Mean demand points per vertiport: {dest_assignments.mean():.1f}")
    add_to_report(f"  Median demand points per vertiport: {dest_assignments.median():.1f}")
    
    # Analysis 6: UAM Demand Coverage
    add_to_report("\n" + "="*60)
    add_to_report("STEP 3: UAM DEMAND COVERAGE ANALYSIS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("From the potential UAM trips calculated in Step 2, we now determine how many")
    add_to_report("can actually be served by the current vertiport network. A UAM trip is COVERED")
    add_to_report("ONLY when BOTH origin AND destination are within 5km of vertiports.")
    add_to_report("")
    add_to_report("COVERAGE DEFINITION:")
    add_to_report("✓ COVERED = Both origin AND destination within 5km catchment area")
    add_to_report("✗ NOT COVERED = At least one endpoint outside 5km catchment area")
    add_to_report("")
    
    # Calculate UAM demand for each coverage category
    uam_both_covered = df[both_covered][uam_prob_col].sum()
    uam_only_origin_covered = df[only_origin_covered][uam_prob_col].sum()
    uam_only_dest_covered = df[only_dest_covered][uam_prob_col].sum()
    uam_neither_covered = df[neither_covered][uam_prob_col].sum()
    
    # Calculate coverage efficiency
    uam_demand_coverage_efficiency = (uam_both_covered / total_potential_uam_trips) * 100
    uam_not_covered = total_potential_uam_trips - uam_both_covered
    
    add_to_report(f"FROM POTENTIAL UAM DEMAND: {total_potential_uam_trips:,.0f} trips")
    add_to_report(f"")
    add_to_report(f"✓ COVERED UAM DEMAND: {uam_both_covered:,.0f} trips ({uam_demand_coverage_efficiency:.1f}%)")
    add_to_report(f"    → CAN be served by current 74 vertiports")
    add_to_report(f"    → Both origin AND destination within 5km")
    add_to_report(f"")
    add_to_report(f"✗ NOT COVERED UAM DEMAND: {uam_not_covered:,.0f} trips ({(100 - uam_demand_coverage_efficiency):.1f}%)")
    add_to_report(f"    → CANNOT be served by current vertiport network")
    add_to_report(f"    → At least one endpoint outside 5km")
    
    # Breakdown of not covered
    uam_partial = uam_only_origin_covered + uam_only_dest_covered
    add_to_report(f"")
    add_to_report(f"    Breakdown of NOT COVERED:")
    add_to_report(f"      - Partial coverage (one endpoint only): {uam_partial:,.0f} trips ({uam_partial/total_potential_uam_trips*100:.1f}%)")
    add_to_report(f"      - No coverage (both endpoints outside): {uam_neither_covered:,.0f} trips ({uam_neither_covered/total_potential_uam_trips*100:.1f}%)")
    
    # Count trips in each category
    both_covered_count = both_covered.sum()
    only_origin_covered_count = only_origin_covered.sum()
    only_dest_covered_count = only_dest_covered.sum()
    neither_covered_count = neither_covered.sum()
    partial_coverage = only_origin_covered_count + only_dest_covered_count
    
    add_to_report(f"")
    add_to_report(f"TRIP COUNTS BY COVERAGE STATUS:")
    add_to_report(f"  ✓ Both endpoints covered: {both_covered_count:,} trips ({both_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  ✗ Only origin covered: {only_origin_covered_count:,} trips ({only_origin_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  ✗ Only destination covered: {only_dest_covered_count:,} trips ({only_dest_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  ✗ Neither endpoint covered: {neither_covered_count:,} trips ({neither_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"")
    add_to_report(f"  Total: {both_covered_count + partial_coverage + neither_covered_count:,} trips")
    
    # Verify math
    uam_total_check = uam_both_covered + uam_partial + uam_neither_covered
    add_to_report(f"")
    add_to_report(f"VERIFICATION:")
    add_to_report(f"  {uam_both_covered:,.0f} (covered) + {uam_not_covered:,.0f} (not covered) = {uam_total_check:,.0f}")
    add_to_report(f"  Should equal potential UAM demand: {total_potential_uam_trips:,.0f} ✓")
    
    # Average UAM probability by coverage category
    avg_uam_prob_both = df[both_covered][uam_prob_col].mean()
    avg_uam_prob_partial = df[only_origin_covered | only_dest_covered][uam_prob_col].mean()
    avg_uam_prob_neither = df[neither_covered][uam_prob_col].mean()
    
    add_to_report(f"")
    add_to_report(f"AVERAGE UAM PROBABILITY BY COVERAGE STATUS:")
    add_to_report(f"  Both endpoints covered: {avg_uam_prob_both:.4f} ({avg_uam_prob_both*100:.2f}%)")
    add_to_report(f"  Partial coverage: {avg_uam_prob_partial:.4f} ({avg_uam_prob_partial*100:.2f}%)")
    add_to_report(f"  Neither endpoint covered: {avg_uam_prob_neither:.4f} ({avg_uam_prob_neither*100:.2f}%)")
    
    # Analysis 7: Catchment Area Parameters
    add_to_report("\n" + "="*60)
    add_to_report("CATCHMENT AREA PARAMETERS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("These parameters define the maximum distance from vertiports")
    add_to_report("where demand points are considered 'covered'.")
    add_to_report("")
    add_to_report(f"Walking catchment distance: 1,000m (1.0km)")
    add_to_report(f"Car catchment distance: 5,000m (5.0km)")
    add_to_report(f"Note: A demand point is covered only if BOTH origin AND destination")
    add_to_report(f"      are within their respective catchment areas.")
    
    # Summary
    add_to_report("\n" + "="*60)
    add_to_report("SUMMARY")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("KEY FINDINGS (3-STEP ANALYSIS):")
    add_to_report("")
    add_to_report(f"STEP 1 - TOTAL DEMAND POINTS:")
    add_to_report(f"  • Total trips: {total_trips:,}")
    add_to_report(f"  • Total demand points: {total_demand_points:,} (1 trip = 1 demand point)")
    
    add_to_report(f"\nSTEP 2 - POTENTIAL UAM TRIPS:")
    add_to_report(f"  • Average UAM probability: {avg_uam_probability:.4f} ({avg_uam_probability*100:.2f}%)")
    add_to_report(f"  • Calculation: {total_trips:,} × {avg_uam_probability:.4f}")
    add_to_report(f"  • Potential UAM trips: {total_potential_uam_trips:,.0f}")
    
    add_to_report(f"\nSTEP 3 - UAM DEMAND COVERAGE:")
    add_to_report(f"  • From {total_potential_uam_trips:,.0f} potential UAM trips:")
    add_to_report(f"      ✓ COVERED: {uam_both_covered:,.0f} trips ({uam_demand_coverage_efficiency:.1f}%)")
    add_to_report(f"         (Both endpoints within 5km - CAN be served)")
    add_to_report(f"      ✗ NOT COVERED: {uam_not_covered:,.0f} trips ({(100-uam_demand_coverage_efficiency):.1f}%)")
    add_to_report(f"         (At least one endpoint outside 5km - CANNOT be served)")
    add_to_report(f"")
    add_to_report(f"  • Coverage efficiency: {uam_demand_coverage_efficiency:.1f}% of potential UAM demand can be served")
    
    # Add header information to the report
    header_info = [
        "VERTIPORT COVERAGE ANALYSIS REPORT",
        "=" * 50,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}",
        "",
        "OVERVIEW:",
        "This report analyzes the coverage of 74 optimized vertiports across",
        "the study area. A 'demand point' represents a complete trip from origin",
        "to destination, and is considered 'covered' only if BOTH endpoints are",
        "within vertiport catchment areas.",
        "",
        "METHODOLOGY:",
        "• Demand points: Complete trips (origin to destination)",
        "• Coverage definition: BOTH endpoints must be within catchment",
        "• UAM demand: Weighted by ML-predicted UAM probability values",
        "• Catchment areas: 1km walking, 5km car access",
        "",
        "=" * 50,
        ""
    ]
    
    # Create coverage summary CSV file
    summary_file = os.path.join(output_dir, "5km_coverage_summary.csv")
    
    try:
        # Prepare summary data
        summary_data = {
            'Metric': [
                'STEP 1 - Total Trips',
                'STEP 1 - Total Demand Points', 
                'STEP 2 - Average UAM Probability',
                'STEP 2 - Average UAM Probability (%)',
                'STEP 2 - Potential UAM Trips',
                'STEP 2 - Calculation Method',
                'STEP 3 - Covered UAM Demand (Both Endpoints)',
                'STEP 3 - Not Covered UAM Demand',
                'STEP 3 - UAM Demand Coverage Efficiency (%)',
                'Trips - Both Endpoints Covered',
                'Trips - Only Origin Covered',
                'Trips - Only Destination Covered',
                'Trips - Neither Endpoint Covered',
                'Origins within 5km (Supplementary)',
                'Destinations within 5km (Supplementary)',
                'UAM Demand - Partial Coverage',
                'UAM Demand - No Coverage'
            ],
            'Value': [
                f"{total_trips:,}",
                f"{total_demand_points:,}",
                f"{avg_uam_probability:.4f}",
                f"{avg_uam_probability*100:.2f}%",
                f"{total_potential_uam_trips:,.0f}",
                f"{total_trips:,} × {avg_uam_probability:.4f}",
                f"{uam_both_covered:,.0f}",
                f"{uam_not_covered:,.0f}",
                f"{uam_demand_coverage_efficiency:.1f}%",
                f"{both_covered_count:,}",
                f"{only_origin_covered_count:,}",
                f"{only_dest_covered_count:,}",
                f"{neither_covered_count:,}",
                f"{origins_in_catchment:,}",
                f"{dests_in_catchment:,}",
                f"{uam_partial:,.0f}",
                f"{uam_neither_covered:,.0f}"
            ]
        }
        
        # Create and save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        # Write the complete report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header_info + report_content))
        
        add_to_report(f"\n" + "="*60)
        add_to_report("FILES SAVED")
        add_to_report("="*60)
        add_to_report(f"Detailed report saved to: {output_file}")
        add_to_report(f"Summary CSV saved to: {summary_file}")
        add_to_report(f"Report file size: {os.path.getsize(output_file):,} bytes")
        add_to_report(f"Summary file size: {os.path.getsize(summary_file):,} bytes")
        
    except Exception as e:
        add_to_report(f"\nError saving files: {e}")
        return
    
    return {
        'total_trips': total_trips,
        'total_demand_points': total_demand_points,
        'avg_uam_probability': avg_uam_probability,
        'total_potential_uam_trips': total_potential_uam_trips,
        'uam_covered': uam_both_covered,
        'uam_not_covered': uam_not_covered,
        'uam_demand_coverage_efficiency': uam_demand_coverage_efficiency,
        'both_covered_count': both_covered_count,
        'only_origin_covered': only_origin_covered_count,
        'only_dest_covered': only_dest_covered_count,
        'partial_coverage': partial_coverage,
        'neither_covered': neither_covered_count,
        'origins_covered': origins_in_catchment,
        'dests_covered': dests_in_catchment,
        'origin_assignments': origin_assignments,
        'dest_assignments': dest_assignments,
        'uam_partial': uam_partial,
        'uam_neither_covered': uam_neither_covered
    }

if __name__ == "__main__":
    results = analyze_vertiport_coverage()
    
    # Print a concise summary to console
    if results:
        print("\n" + "="*80)
        print("COVERAGE ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nSTEP 1 - TOTAL DEMAND POINTS:")
        print(f"  Total trips = Total demand points: {results['total_trips']:,}")
        print(f"  (Each trip from origin to destination = 1 demand point)")
        
        print(f"\nSTEP 2 - POTENTIAL UAM TRIPS:")
        print(f"  Average UAM probability: {results['avg_uam_probability']:.4f} ({results['avg_uam_probability']*100:.2f}%)")
        print(f"  Calculation: {results['total_trips']:,} × {results['avg_uam_probability']:.4f}")
        print(f"  Potential UAM trips: {results['total_potential_uam_trips']:,.0f}")
        
        print(f"\nSTEP 3 - UAM DEMAND COVERAGE:")
        print(f"  From {results['total_potential_uam_trips']:,.0f} potential UAM trips:")
        print(f"    ✓ COVERED: {results['uam_covered']:,.0f} trips ({results['uam_demand_coverage_efficiency']:.1f}%)")
        print(f"       → CAN be served (both endpoints within 5km)")
        print(f"    ✗ NOT COVERED: {results['uam_not_covered']:,.0f} trips ({100-results['uam_demand_coverage_efficiency']:.1f}%)")
        print(f"       → CANNOT be served (at least one endpoint outside 5km)")
        
        print("\n" + "="*80)
        print(f"COVERAGE EFFICIENCY: {results['uam_demand_coverage_efficiency']:.1f}% of potential UAM demand can be served")
        print("="*80)
        print("\nDetailed report and summary CSV saved to Coverage_Analysis directory")
        print("="*80)
