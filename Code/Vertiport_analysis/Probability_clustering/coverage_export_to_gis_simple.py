import pandas as pd
import numpy as np
import os

def create_gis_export_simple():
    """
    Create GIS-compatible exports using simple CSV format
    """
    print("Creating GIS exports for vertiport coverage analysis...")
    
    # Load the prediction results
    try:
        print("Loading prediction data...")
        df = pd.read_csv("D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv", 
                        low_memory=False)
        print(f"Loaded {len(df):,} trips")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load vertiport coordinates
    try:
        vertiports = pd.read_csv("D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_final.csv")
        print(f"Loaded {len(vertiports)} vertiports")
    except Exception as e:
        print(f"Error loading vertiport coordinates: {e}")
        return
    
    # Create output directory
    output_dir = "/Result/Vertiport_analysis/Probability_clustering/Coverage_GIS_Export"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. CREATE VERTIPORT POINTS (CSV format)
    print("\n1. Creating vertiport points...")
    
    vertiport_export = vertiports.copy()
    vertiport_export['vertiport_id'] = range(1, len(vertiports) + 1)
    vertiport_export['vertiport_name'] = [f'Vertiport_{i}' for i in range(1, len(vertiports) + 1)]
    vertiport_export['type'] = 'vertiport'
    
    # Save vertiport points
    vertiport_output = os.path.join(output_dir, "vertiports.csv")
    vertiport_export.to_csv(vertiport_output, index=False)
    print(f"Vertiport points saved: {vertiport_output}")
    
    # 2. CREATE DEMAND POINT SAMPLES
    print("\n2. Creating demand point samples...")
    
    # Sample demand points for visualization (too many to show all)
    sample_size = min(20000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Create origin points
    origin_points = pd.DataFrame({
        'point_id': range(1, len(df_sample) + 1),
        'X': df_sample['originX'],
        'Y': df_sample['originY'],
        'point_type': 'origin',
        'in_catchment': df_sample['origin_in_catchment'],
        'access_mode': df_sample['origin_access_mode'],
        'distance_to_vertiport': df_sample['origin_to_vertiport_dist'],
        'uam_probability': df_sample['prob_mode_Autonomous Flying Taxi'],
        'trip_id': df_sample.index
    })
    
    # Create destination points
    dest_points = pd.DataFrame({
        'point_id': range(len(df_sample) + 1, len(df_sample) * 2 + 1),
        'X': df_sample['destinationX'],
        'Y': df_sample['destinationY'],
        'point_type': 'destination',
        'in_catchment': df_sample['dest_in_catchment'],
        'access_mode': df_sample['dest_access_mode'],
        'distance_to_vertiport': df_sample['dest_to_vertiport_dist'],
        'uam_probability': df_sample['prob_mode_Autonomous Flying Taxi'],
        'trip_id': df_sample.index
    })
    
    # Combine origin and destination points
    demand_points = pd.concat([origin_points, dest_points], ignore_index=True)
    
    # Save demand points
    demand_output = os.path.join(output_dir, "demand_points_sample.csv")
    demand_points.to_csv(demand_output, index=False)
    print(f"Demand points sample saved: {demand_output}")
    
    # 3. CREATE VERTIPORT ASSIGNMENT ANALYSIS
    print("\n3. Creating vertiport assignment analysis...")
    
    # Count demand points assigned to each vertiport
    origin_assignments = df['uam_origin_vertiport'].value_counts().sort_index()
    dest_assignments = df['uam_dest_vertiport'].value_counts().sort_index()
    
    # Add assignment statistics to vertiport data
    vertiport_stats = vertiport_export.copy()
    vertiport_stats['origin_assignments'] = vertiport_stats['vertiport_id'].map(origin_assignments).fillna(0)
    vertiport_stats['destination_assignments'] = vertiport_stats['vertiport_id'].map(dest_assignments).fillna(0)
    vertiport_stats['total_assignments'] = vertiport_stats['origin_assignments'] + vertiport_stats['destination_assignments']
    
    # Calculate assignment density (approximate)
    vertiport_stats['assignment_density'] = vertiport_stats['total_assignments'] / (np.pi * 5**2)  # per km² (5km radius)
    
    # Save vertiport with assignment statistics
    vertiport_stats_output = os.path.join(output_dir, "vertiports_with_assignments.csv")
    vertiport_stats.to_csv(vertiport_stats_output, index=False)
    print(f"Vertiports with assignment statistics saved: {vertiport_stats_output}")
    
    # 4. CREATE COVERAGE ANALYSIS SUMMARY
    print("\n4. Creating coverage analysis summary...")
    
    # Calculate coverage statistics
    total_trips = len(df)
    total_demand_points = total_trips * 2
    
    origins_covered = df['origin_in_catchment'].sum()
    dests_covered = df['dest_in_catchment'].sum()
    total_covered = origins_covered + dests_covered
    
    both_covered = (df['origin_in_catchment'] & df['dest_in_catchment']).sum()
    either_covered = (df['origin_in_catchment'] | df['dest_in_catchment']).sum()
    neither_covered = total_trips - either_covered
    
    # UAM trip calculations
    uam_prob_col = 'prob_mode_Autonomous Flying Taxi'
    if uam_prob_col in df.columns:
        mean_uam_prob = df[uam_prob_col].mean()
        total_potential_uam = total_trips * mean_uam_prob
        uam_either_covered = either_covered * mean_uam_prob
        uam_both_covered = both_covered * mean_uam_prob
        uam_neither_covered = neither_covered * mean_uam_prob
    else:
        mean_uam_prob = 0.2969  # From method report
        total_potential_uam = total_trips * mean_uam_prob
        uam_either_covered = either_covered * mean_uam_prob
        uam_both_covered = both_covered * mean_uam_prob
        uam_neither_covered = neither_covered * mean_uam_prob
    
    # Create summary statistics file
    summary_stats = {
        'Metric': [
            'Total Trips',
            'Total Demand Points',
            'Origins in Catchment',
            'Destinations in Catchment', 
            'Total Demand Points Covered',
            'Coverage Percentage',
            'Trips Both Endpoints Covered',
            'Trips Either Endpoint Covered',
            'Trips Neither Endpoint Covered',
            'Mean UAM Probability',
            'Total Potential UAM Trips',
            'UAM Trips Either Endpoint Covered',
            'UAM Trips Both Endpoints Covered',
            'UAM Trips Neither Endpoint Covered',
            'UAM Coverage Efficiency (%)'
        ],
        'Value': [
            total_trips,
            total_demand_points,
            origins_covered,
            dests_covered,
            total_covered,
            f"{total_covered/total_demand_points*100:.1f}%",
            both_covered,
            either_covered,
            neither_covered,
            f"{mean_uam_prob:.4f}",
            f"{total_potential_uam:,.0f}",
            f"{uam_either_covered:,.0f}",
            f"{uam_both_covered:,.0f}",
            f"{uam_neither_covered:,.0f}",
            f"{uam_either_covered/total_potential_uam*100:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_output = os.path.join(output_dir, "coverage_summary.csv")
    summary_df.to_csv(summary_output, index=False)
    print(f"Coverage summary saved: {summary_output}")
    
    # 5. CREATE CATCHMENT AREA COORDINATES (for manual polygon creation)
    print("\n5. Creating catchment area coordinates...")
    
    # Create 5km buffer coordinates for each vertiport (simplified as circles)
    catchment_data = []
    for i, (_, vertiport) in enumerate(vertiports.iterrows()):
        x, y = vertiport['X'], vertiport['Y']
        
        # Create circle coordinates (simplified - 36 points)
        angles = np.linspace(0, 2*np.pi, 37)  # 37 points to close the circle
        circle_x = x + 5000 * np.cos(angles)
        circle_y = y + 5000 * np.sin(angles)
        
        # Create polygon coordinates
        polygon_coords = list(zip(circle_x, circle_y))
        
        catchment_data.append({
            'vertiport_id': i + 1,
            'catchment_type': 'car_catchment_5km',
            'radius': 5000,
            'polygon_coords': str(polygon_coords)
        })
        
        # Also create 1km walking catchment
        circle_x_walk = x + 1000 * np.cos(angles)
        circle_y_walk = y + 1000 * np.sin(angles)
        polygon_coords_walk = list(zip(circle_x_walk, circle_y_walk))
        
        catchment_data.append({
            'vertiport_id': i + 1,
            'catchment_type': 'walking_catchment_1km',
            'radius': 1000,
            'polygon_coords': str(polygon_coords_walk)
        })
    
    catchment_df = pd.DataFrame(catchment_data)
    catchment_output = os.path.join(output_dir, "catchment_areas.csv")
    catchment_df.to_csv(catchment_output, index=False)
    print(f"Catchment areas coordinates saved: {catchment_output}")
    
    # 6. CREATE README FILE
    print("\n6. Creating documentation...")
    
    readme_content = f"""
VERTIPORT COVERAGE ANALYSIS - GIS EXPORTS
==========================================

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

FILES CREATED:
1. vertiports.csv - Vertiport locations (74 points) with X,Y coordinates
2. vertiports_with_assignments.csv - Vertiports with demand assignment statistics
3. demand_points_sample.csv - Sample of demand points (origins and destinations)
4. catchment_areas.csv - Catchment area polygon coordinates
5. coverage_summary.csv - Summary statistics

COORDINATE SYSTEM: Local coordinate system (meters)
UNITS: Meters

COVERAGE STATISTICS:
- Total Demand Points: {total_demand_points:,}
- Demand Points Covered: {total_covered:,} ({total_covered/total_demand_points*100:.1f}%)
- UAM Coverage Efficiency: {uam_either_covered/total_potential_uam*100:.1f}%

CATCHMENT AREAS:
- Car Catchment: 5,000m radius
- Walking Catchment: 1,000m radius

HOW TO USE IN GIS SOFTWARE:

1. IMPORT CSV FILES:
   - Import vertiports.csv as point layer
   - Import demand_points_sample.csv as point layer
   - Use X, Y columns for coordinates

2. CREATE CATCHMENT POLYGONS:
   - Use catchment_areas.csv to create buffer polygons
   - Or create 5km/1km buffers around vertiport points

3. VISUALIZATION:
   - Color-code demand points by 'in_catchment' field
   - Use 'access_mode' field to distinguish walking vs car access
   - Use 'uam_probability' for UAM demand visualization

4. ANALYSIS:
   - Overlay demand points with catchment areas
   - Calculate coverage statistics by area
   - Analyze spatial patterns in UAM demand

For detailed analysis, see coverage_summary.csv
"""
    
    readme_output = os.path.join(output_dir, "README.txt")
    with open(readme_output, 'w') as f:
        f.write(readme_content)
    print(f"Documentation saved: {readme_output}")
    
    print(f"\n✓ All GIS exports completed successfully!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Files ready for import into ArcGIS, QGIS, or other GIS software")
    print(f"✓ CSV format can be easily converted to shapefiles")
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_gis_export_simple()
