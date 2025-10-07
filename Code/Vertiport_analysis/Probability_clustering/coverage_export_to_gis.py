import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

def create_vertiport_coverage_gis():
    """
    Export vertiport coverage analysis to GIS formats for visualization
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
    
    # 1. CREATE VERTIPORT POINTS
    print("\n1. Creating vertiport points...")
    
    # Create Point geometries for vertiports
    vertiport_geometry = [Point(x, y) for x, y in zip(vertiports['X'], vertiports['Y'])]
    vertiport_gdf = gpd.GeoDataFrame(vertiports, geometry=vertiport_geometry)
    
    # Add additional attributes
    vertiport_gdf['vertiport_id'] = range(1, len(vertiports) + 1)
    vertiport_gdf['vertiport_name'] = [f'Vertiport_{i}' for i in range(1, len(vertiports) + 1)]
    vertiport_gdf['type'] = 'vertiport'
    
    # Set CRS (assuming UTM or local coordinate system in meters)
    vertiport_gdf.crs = "EPSG:32632"  # UTM Zone 32N (adjust if needed)
    
    # Save vertiport points
    vertiport_output = os.path.join(output_dir, "vertiports.shp")
    vertiport_gdf.to_file(vertiport_output)
    print(f"Vertiport points saved: {vertiport_output}")
    
    # 2. CREATE VERTIPORT CATCHMENT AREAS (BUFFERS)
    print("\n2. Creating vertiport catchment areas...")
    
    # Create 5km buffers around vertiports (car catchment)
    catchment_gdf = vertiport_gdf.copy()
    catchment_gdf['geometry'] = vertiport_gdf.geometry.buffer(5000)  # 5000m = 5km
    catchment_gdf['catchment_type'] = 'car_catchment_5km'
    catchment_gdf['catchment_radius'] = 5000
    
    # Create 1km buffers (walking catchment)
    walking_catchment_gdf = vertiport_gdf.copy()
    walking_catchment_gdf['geometry'] = vertiport_gdf.geometry.buffer(1000)  # 1000m = 1km
    walking_catchment_gdf['catchment_type'] = 'walking_catchment_1km'
    walking_catchment_gdf['catchment_radius'] = 1000
    
    # Combine both catchment types
    all_catchments = pd.concat([catchment_gdf, walking_catchment_gdf], ignore_index=True)
    
    # Save catchment areas
    catchment_output = os.path.join(output_dir, "vertiport_catchments.shp")
    all_catchments.to_file(catchment_output)
    print(f"Catchment areas saved: {catchment_output}")
    
    # 3. CREATE DEMAND POINT SAMPLES
    print("\n3. Creating demand point samples...")
    
    # Sample demand points for visualization (too many to show all)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Create origin points
    origin_geometry = [Point(x, y) for x, y in zip(df_sample['originX'], df_sample['originY'])]
    origin_gdf = gpd.GeoDataFrame(df_sample, geometry=origin_geometry)
    origin_gdf['point_type'] = 'origin'
    origin_gdf['in_catchment'] = origin_gdf['origin_in_catchment']
    origin_gdf['access_mode'] = origin_gdf['origin_access_mode']
    origin_gdf['distance_to_vertiport'] = origin_gdf['origin_to_vertiport_dist']
    
    # Create destination points
    dest_geometry = [Point(x, y) for x, y in zip(df_sample['destinationX'], df_sample['destinationY'])]
    dest_gdf = gpd.GeoDataFrame(df_sample, geometry=dest_geometry)
    dest_gdf['point_type'] = 'destination'
    dest_gdf['in_catchment'] = dest_gdf['dest_in_catchment']
    dest_gdf['access_mode'] = dest_gdf['dest_access_mode']
    dest_gdf['distance_to_vertiport'] = dest_gdf['dest_to_vertiport_dist']
    
    # Combine origin and destination points
    demand_points = pd.concat([origin_gdf, dest_gdf], ignore_index=True)
    
    # Select relevant columns for GIS
    demand_columns = ['geometry', 'point_type', 'in_catchment', 'access_mode', 
                     'distance_to_vertiport', 'prob_mode_Autonomous Flying Taxi']
    demand_gdf = demand_points[demand_columns].copy()
    
    # Set CRS
    demand_gdf.crs = "EPSG:32632"
    
    # Save demand points
    demand_output = os.path.join(output_dir, "demand_points_sample.shp")
    demand_gdf.to_file(demand_output)
    print(f"Demand points sample saved: {demand_output}")
    
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
    
    # 5. CREATE VERTIPORT ASSIGNMENT ANALYSIS
    print("\n5. Creating vertiport assignment analysis...")
    
    # Count demand points assigned to each vertiport
    origin_assignments = df['uam_origin_vertiport'].value_counts().sort_index()
    dest_assignments = df['uam_dest_vertiport'].value_counts().sort_index()
    
    # Add assignment statistics to vertiport data
    vertiport_stats = vertiport_gdf.copy()
    vertiport_stats['origin_assignments'] = vertiport_stats['vertiport_id'].map(origin_assignments).fillna(0)
    vertiport_stats['destination_assignments'] = vertiport_stats['vertiport_id'].map(dest_assignments).fillna(0)
    vertiport_stats['total_assignments'] = vertiport_stats['origin_assignments'] + vertiport_stats['destination_assignments']
    vertiport_stats['assignment_density'] = vertiport_stats['total_assignments'] / vertiport_stats.geometry.buffer(5000).area * 1000000  # per km²
    
    # Save vertiport with assignment statistics
    vertiport_stats_output = os.path.join(output_dir, "vertiports_with_assignments.shp")
    vertiport_stats.to_file(vertiport_stats_output)
    print(f"Vertiports with assignment statistics saved: {vertiport_stats_output}")
    
    # 6. CREATE README FILE
    print("\n6. Creating documentation...")
    
    readme_content = f"""
VERTIPORT COVERAGE ANALYSIS - GIS EXPORTS
==========================================

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

FILES CREATED:
1. vertiports.shp - Vertiport locations (74 points)
2. vertiport_catchments.shp - Catchment areas around vertiports (5km car, 1km walking)
3. demand_points_sample.shp - Sample of demand points (origins and destinations)
4. vertiports_with_assignments.shp - Vertiports with demand assignment statistics
5. coverage_summary.csv - Summary statistics

COORDINATE SYSTEM: EPSG:32632 (UTM Zone 32N)
UNITS: Meters

COVERAGE STATISTICS:
- Total Demand Points: {total_demand_points:,}
- Demand Points Covered: {total_covered:,} ({total_covered/total_demand_points*100:.1f}%)
- UAM Coverage Efficiency: {uam_either_covered/total_potential_uam*100:.1f}%

CATCHMENT AREAS:
- Car Catchment: 5,000m radius
- Walking Catchment: 1,000m radius

VISUALIZATION SUGGESTIONS:
1. Use vertiports.shp as point layer
2. Use vertiport_catchments.shp for coverage areas
3. Use demand_points_sample.shp for demand visualization
4. Color-code by 'in_catchment' field for coverage analysis
5. Use 'access_mode' field to distinguish walking vs car access

For detailed analysis, see coverage_summary.csv
"""
    
    readme_output = os.path.join(output_dir, "README.txt")
    with open(readme_output, 'w') as f:
        f.write(readme_content)
    print(f"Documentation saved: {readme_output}")
    
    print(f"\n✓ All GIS exports completed successfully!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Files ready for import into ArcGIS, QGIS, or other GIS software")
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_vertiport_coverage_gis()
