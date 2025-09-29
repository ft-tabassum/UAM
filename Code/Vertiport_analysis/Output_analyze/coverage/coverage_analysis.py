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
    output_dir = "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Coverage_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    output_file = os.path.join(output_dir, "coverage_analysis_report.txt")
    
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
            "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv",
            low_memory=False)
        add_to_report(f"Loaded {len(df):,} demand points")
        
    except Exception as e:
        add_to_report(f"Error loading data: {e}")
        return
    
    # Load vertiport coordinates
    try:
        vertiports = pd.read_csv(
            "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_final.csv")
        add_to_report(f"Loaded {len(vertiports)} vertiports")
    except Exception as e:
        add_to_report(f"Error loading vertiport coordinates: {e}")
        return
    
    # Analysis 1: Individual Demand Point Coverage
    add_to_report("\n" + "="*60)
    add_to_report("INDIVIDUAL DEMAND POINT COVERAGE ANALYSIS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("This analysis counts individual origin and destination points (demand points)")
    add_to_report("and determines how many are within the catchment area of vertiports.")
    add_to_report("Each trip has 2 demand points: 1 origin + 1 destination.")
    add_to_report("")
    
    # Count individual demand points within catchment areas
    total_trips = len(df)
    total_demand_points = total_trips * 2  # origins + destinations
    
    origins_in_catchment = df['origin_in_catchment'].sum()
    dests_in_catchment = df['dest_in_catchment'].sum()
    total_demand_points_covered = origins_in_catchment + dests_in_catchment
    
    origins_outside_catchment = total_trips - origins_in_catchment
    dests_outside_catchment = total_trips - dests_in_catchment
    total_demand_points_outside = origins_outside_catchment + dests_outside_catchment
    
    add_to_report(f"Total trips: {total_trips:,}")
    add_to_report(f"Total individual demand points (origins + destinations): {total_demand_points:,}")
    add_to_report(f"\nINDIVIDUAL DEMAND POINT COVERAGE:")
    add_to_report(f"  Total demand points within catchment area: {total_demand_points_covered:,} ({total_demand_points_covered/total_demand_points*100:.1f}%)")
    add_to_report(f"  Total demand points outside catchment area: {total_demand_points_outside:,} ({total_demand_points_outside/total_demand_points*100:.1f}%)")
    
    add_to_report(f"\nBREAKDOWN BY TYPE:")
    add_to_report(f"  Origin points within catchment: {origins_in_catchment:,} ({origins_in_catchment/total_trips*100:.1f}% of origins)")
    add_to_report(f"  Destination points within catchment: {dests_in_catchment:,} ({dests_in_catchment/total_trips*100:.1f}% of destinations)")
    add_to_report(f"  Origin points outside catchment: {origins_outside_catchment:,} ({origins_outside_catchment/total_trips*100:.1f}% of origins)")
    add_to_report(f"  Destination points outside catchment: {dests_outside_catchment:,} ({dests_outside_catchment/total_trips*100:.1f}% of destinations)")
    
    # Analysis 2: Access Mode Distribution
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
    
    # Analysis 3: Distance Statistics
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
    
    # Analysis 4: Vertiport Assignment Distribution
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
    
    # Analysis 5: Complete Trip Coverage Analysis
    add_to_report("\n" + "="*60)
    add_to_report("COMPLETE TRIP COVERAGE ANALYSIS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("This analysis categorizes trips based on whether their origin and/or")
    add_to_report("destination points are within vertiport catchment areas.")
    add_to_report("Categories are mutually exclusive (no overlap).")
    add_to_report("")
    
    # Create mutually exclusive trip categories
    both_covered = df['origin_in_catchment'] & df['dest_in_catchment']
    only_origin_covered = df['origin_in_catchment'] & ~df['dest_in_catchment']
    only_dest_covered = ~df['origin_in_catchment'] & df['dest_in_catchment']
    neither_covered = ~df['origin_in_catchment'] & ~df['dest_in_catchment']
    
    both_covered_count = both_covered.sum()
    only_origin_covered_count = only_origin_covered.sum()
    only_dest_covered_count = only_dest_covered.sum()
    neither_covered_count = neither_covered.sum()
    
    # Verify the math adds up
    total_check = both_covered_count + only_origin_covered_count + only_dest_covered_count + neither_covered_count
    
    add_to_report(f"COMPLETE TRIP COVERAGE (Mutually Exclusive Categories):")
    add_to_report(f"  Both origin and destination covered: {both_covered_count:,} ({both_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  Only origin covered: {only_origin_covered_count:,} ({only_origin_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  Only destination covered: {only_dest_covered_count:,} ({only_dest_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  Neither origin nor destination covered: {neither_covered_count:,} ({neither_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  Total verification: {total_check:,} (should equal {total_trips:,})")
    
    # Also show the overlapping categories for reference
    either_covered = df['origin_in_catchment'] | df['dest_in_catchment']
    either_covered_count = either_covered.sum()
    add_to_report(f"\nREFERENCE (Overlapping Categories):")
    add_to_report(f"  At least one endpoint covered: {either_covered_count:,} ({either_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"  (This includes 'both covered' trips)")
    
    # Analysis 5.1: UAM Trip Coverage Analysis
    add_to_report("\n" + "="*60)
    add_to_report("UAM TRIP COVERAGE ANALYSIS")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("DESCRIPTION:")
    add_to_report("This analysis weights trip coverage by UAM probability values.")
    add_to_report("Each trip has a probability (0-1) of using UAM based on ML predictions.")
    add_to_report("Total potential UAM trips = sum of all probability values.")
    add_to_report("This shows how much actual UAM demand is covered by vertiports.")
    add_to_report("")
    
    # Find UAM probability column
    uam_prob_col = 'prob_mode_Autonomous Flying Taxi'
    
    # Verify the column exists
    if uam_prob_col not in df.columns:
        add_to_report(f"Error: UAM probability column '{uam_prob_col}' not found in data")
        prob_cols = [col for col in df.columns if 'prob_mode_' in col]
        add_to_report(f"Available probability columns: {prob_cols}")
        return
    
    if uam_prob_col is not None:
        add_to_report(f"Using UAM probability column: {uam_prob_col}")
        
        # Calculate UAM trips within catchment areas using mutually exclusive categories
        uam_probs = df[uam_prob_col]
        
        # UAM trips for each mutually exclusive category
        uam_both_covered = df[both_covered][uam_prob_col].sum()
        uam_only_origin_covered = df[only_origin_covered][uam_prob_col].sum()
        uam_only_dest_covered = df[only_dest_covered][uam_prob_col].sum()
        uam_neither_covered = df[neither_covered][uam_prob_col].sum()
        
        # Total potential UAM trips
        total_potential_uam_trips = uam_probs.sum()
        
        # Verify the math adds up
        uam_total_check = uam_both_covered + uam_only_origin_covered + uam_only_dest_covered + uam_neither_covered
        
        add_to_report(f"UAM TRIP COVERAGE (Mutually Exclusive Categories):")
        add_to_report(f"  Total potential UAM trips: {total_potential_uam_trips:,.0f}")
        add_to_report(f"  UAM trips with both endpoints covered: {uam_both_covered:,.0f} ({uam_both_covered/total_potential_uam_trips*100:.1f}% of potential UAM)")
        add_to_report(f"  UAM trips with only origin covered: {uam_only_origin_covered:,.0f} ({uam_only_origin_covered/total_potential_uam_trips*100:.1f}% of potential UAM)")
        add_to_report(f"  UAM trips with only destination covered: {uam_only_dest_covered:,.0f} ({uam_only_dest_covered/total_potential_uam_trips*100:.1f}% of potential UAM)")
        add_to_report(f"  UAM trips with neither endpoint covered: {uam_neither_covered:,.0f} ({uam_neither_covered/total_potential_uam_trips*100:.1f}% of potential UAM)")
        add_to_report(f"  Total verification: {uam_total_check:,.0f} (should equal {total_potential_uam_trips:,.0f})")
        
        # Also show the overlapping categories for reference
        uam_either_covered = df[either_covered][uam_prob_col].sum()
        uam_coverage_efficiency = (uam_either_covered / total_potential_uam_trips) * 100
        add_to_report(f"\nREFERENCE (Overlapping Categories):")
        add_to_report(f"  UAM trips with at least one endpoint covered: {uam_either_covered:,.0f} ({uam_coverage_efficiency:.1f}% of potential UAM)")
        add_to_report(f"  (This includes 'both covered' trips)")
        
        # Average UAM probability for different coverage scenarios
        avg_uam_prob_both = df[both_covered][uam_prob_col].mean()
        avg_uam_prob_only_origin = df[only_origin_covered][uam_prob_col].mean()
        avg_uam_prob_only_dest = df[only_dest_covered][uam_prob_col].mean()
        avg_uam_prob_neither = df[neither_covered][uam_prob_col].mean()
        avg_uam_prob_either = df[either_covered][uam_prob_col].mean()
        
        add_to_report(f"\nAVERAGE UAM PROBABILITY BY COVERAGE:")
        add_to_report(f"  Both endpoints covered: {avg_uam_prob_both:.4f} (avg UAM probability)")
        add_to_report(f"  Only origin covered: {avg_uam_prob_only_origin:.4f} (avg UAM probability)")
        add_to_report(f"  Only destination covered: {avg_uam_prob_only_dest:.4f} (avg UAM probability)")
        add_to_report(f"  Neither endpoint covered: {avg_uam_prob_neither:.4f} (avg UAM probability)")
        add_to_report(f"  At least one endpoint covered: {avg_uam_prob_either:.4f} (avg UAM probability)")
        
    else:
        add_to_report("UAM probability column not found. Available columns:")
        prob_cols = [col for col in df.columns if 'prob_mode_' in col]
        for col in prob_cols:
            add_to_report(f"  - {col}")
        add_to_report("\nNote: UAM trip analysis requires a UAM probability column")
    
    # Analysis 6: Catchment Area Parameters
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
    add_to_report(f"Note: Car catchment distance is used as the maximum coverage area")
    
    # Summary
    add_to_report("\n" + "="*60)
    add_to_report("SUMMARY")
    add_to_report("="*60)
    add_to_report("")
    add_to_report("KEY FINDINGS:")
    add_to_report("")
    add_to_report(f"INDIVIDUAL DEMAND POINTS:")
    add_to_report(f"• Total individual demand points (origins + destinations): {total_demand_points:,}")
    add_to_report(f"• Demand points within catchment area: {total_demand_points_covered:,} ({total_demand_points_covered/total_demand_points*100:.1f}%)")
    add_to_report(f"• Demand points outside catchment area: {total_demand_points_outside:,} ({total_demand_points_outside/total_demand_points*100:.1f}%)")
    
    add_to_report(f"\nTRIP COVERAGE (Mutually Exclusive Categories):")
    add_to_report(f"• Total trips: {total_trips:,}")
    add_to_report(f"• Trips with both endpoints covered: {both_covered_count:,} ({both_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"• Trips with only origin covered: {only_origin_covered_count:,} ({only_origin_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"• Trips with only destination covered: {only_dest_covered_count:,} ({only_dest_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"• Trips with neither endpoint covered: {neither_covered_count:,} ({neither_covered_count/total_trips*100:.1f}%)")
    add_to_report(f"• Reference - Trips with at least one endpoint covered: {either_covered_count:,} ({either_covered_count/total_trips*100:.1f}%)")
    
    # Add UAM summary if UAM column found
    if uam_prob_col is not None:
        add_to_report(f"\nUAM TRIP COVERAGE (Mutually Exclusive Categories):")
        add_to_report(f"• Total potential UAM trips: {total_potential_uam_trips:,.0f}")
        add_to_report(f"• UAM trips with both endpoints covered: {uam_both_covered:,.0f} ({uam_both_covered/total_potential_uam_trips*100:.1f}%)")
        add_to_report(f"• UAM trips with only origin covered: {uam_only_origin_covered:,.0f} ({uam_only_origin_covered/total_potential_uam_trips*100:.1f}%)")
        add_to_report(f"• UAM trips with only destination covered: {uam_only_dest_covered:,.0f} ({uam_only_dest_covered/total_potential_uam_trips*100:.1f}%)")
        add_to_report(f"• UAM trips with neither endpoint covered: {uam_neither_covered:,.0f} ({uam_neither_covered/total_potential_uam_trips*100:.1f}%)")
        add_to_report(f"• Reference - UAM trips with at least one endpoint covered: {uam_either_covered:,.0f} ({uam_coverage_efficiency:.1f}%)")
    
    # Add header information to the report
    header_info = [
        "VERTIPORT COVERAGE ANALYSIS REPORT",
        "=" * 50,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}",
        "",
        "OVERVIEW:",
        "This report analyzes the coverage of 74 optimized vertiports across",
        "the study area, examining both individual demand point coverage and",
        "UAM trip coverage weighted by probability values.",
        "",
        "METHODOLOGY:",
        "• Individual demand points: Count of origin/destination locations within catchment",
        "• Trip coverage: Categorization of trips based on endpoint coverage",
        "• UAM coverage: Weighted by ML-predicted UAM probability values",
        "• Catchment areas: 1km walking, 5km car access",
        "",
        "=" * 50,
        ""
    ]
    
    # Create coverage summary CSV file
    summary_file = os.path.join(output_dir, "coverage_summary.csv")
    
    try:
        # Prepare summary data
        summary_data = {
            'Metric': [
                'Total Trips',
                'Total Demand Points', 
                'Origins in Catchment',
                'Destinations in Catchment',
                'Total Demand Points Covered',
                'Coverage Percentage',
                'Trips Both Endpoints Covered',
                'Trips Only Origin Covered',
                'Trips Only Destination Covered',
                'Trips Neither Endpoint Covered',
                'Trips Either Endpoint Covered (Reference)',
                'Mean UAM Probability',
                'Total Potential UAM Trips',
                'UAM Trips Both Endpoints Covered',
                'UAM Trips Only Origin Covered',
                'UAM Trips Only Destination Covered',
                'UAM Trips Neither Endpoint Covered',
                'UAM Trips Either Endpoint Covered (Reference)',
                'UAM Coverage Efficiency (%)'
            ],
            'Value': [
                f"{total_trips:,}",
                f"{total_demand_points:,}",
                f"{origins_in_catchment:,}",
                f"{dests_in_catchment:,}",
                f"{total_demand_points_covered:,}",
                f"{total_demand_points_covered/total_demand_points*100:.1f}%",
                f"{both_covered_count:,}",
                f"{only_origin_covered_count:,}",
                f"{only_dest_covered_count:,}",
                f"{neither_covered_count:,}",
                f"{either_covered_count:,}",
                f"{uam_probs.mean():.4f}" if uam_prob_col else "N/A",
                f"{total_potential_uam_trips:,.0f}" if uam_prob_col else "N/A",
                f"{uam_both_covered:,.0f}" if uam_prob_col else "N/A",
                f"{uam_only_origin_covered:,.0f}" if uam_prob_col else "N/A",
                f"{uam_only_dest_covered:,.0f}" if uam_prob_col else "N/A",
                f"{uam_neither_covered:,.0f}" if uam_prob_col else "N/A",
                f"{uam_either_covered:,.0f}" if uam_prob_col else "N/A",
                f"{uam_coverage_efficiency:.1f}%" if uam_prob_col else "N/A"
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
        'total_demand_points_covered': total_demand_points_covered,
        'total_demand_points_outside': total_demand_points_outside,
        'both_covered': both_covered_count,
        'only_origin_covered': only_origin_covered_count,
        'only_dest_covered': only_dest_covered_count,
        'neither_covered': neither_covered_count,
        'either_covered': either_covered_count,  # Reference category
        'origins_covered': origins_in_catchment,
        'dests_covered': dests_in_catchment,
        'origin_assignments': origin_assignments,
        'dest_assignments': dest_assignments,
        'uam_prob_col': uam_prob_col,
        'total_potential_uam_trips': total_potential_uam_trips if uam_prob_col else None,
        'uam_both_covered': uam_both_covered if uam_prob_col else None,
        'uam_only_origin_covered': uam_only_origin_covered if uam_prob_col else None,
        'uam_only_dest_covered': uam_only_dest_covered if uam_prob_col else None,
        'uam_neither_covered': uam_neither_covered if uam_prob_col else None,
        'uam_either_covered': uam_either_covered if uam_prob_col else None,  # Reference category
        'uam_coverage_efficiency': uam_coverage_efficiency if uam_prob_col else None
    }

if __name__ == "__main__":
    results = analyze_vertiport_coverage()
    
    # Print a concise summary to console
    if results:
        print("\n" + "="*80)
        print("COVERAGE ANALYSIS SUMMARY")
        print("="*80)
        print(f"DEMAND POINT COVERAGE: {results['total_demand_points_covered']:,} / {results['total_demand_points']:,} ({results['total_demand_points_covered']/results['total_demand_points']*100:.1f}%)")
        print(f"UAM COVERAGE EFFICIENCY: {results['uam_coverage_efficiency']:.1f}%" if results['uam_coverage_efficiency'] else "UAM COVERAGE: Not available")
        print(f"TOTAL TRIPS: {results['total_trips']:,}")
        print(f"TRIPS WITH BOTH ENDPOINTS COVERED: {results['both_covered']:,} ({results['both_covered']/results['total_trips']*100:.1f}%)")
        print(f"TRIPS WITH NEITHER ENDPOINT COVERED: {results['neither_covered']:,} ({results['neither_covered']/results['total_trips']*100:.1f}%)")
        print("="*80)
        print("Detailed report and summary CSV saved to Coverage_Analysis directory")
        print("="*80)
