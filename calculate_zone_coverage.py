#!/usr/bin/env python3
"""
Calculate zone coverage: Which zones are within 5km of vertiports?
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def calculate_zone_coverage():
    """
    Calculate which zones are covered (within 5km of any vertiport)
    """
    
    print("🚁 Calculating Zone Coverage...")
    
    # Load data
    zones_file = "Result/Vertiport_analysis/Probability_clustering/Centroid/zonePlz_with_aggregated_data.csv"
    vertiports_file = "Result/Vertiport_analysis/Probability_clustering/Centroid/5km_radius_optimized_vertiport_coords_final.csv"
    
    zones_df = pd.read_csv(zones_file)
    vertiports_df = pd.read_csv(vertiports_file)
    
    print(f"📊 Loaded {len(zones_df)} zones and {len(vertiports_df)} vertiports")
    
    # Get coordinates
    zone_coords = zones_df[['CENTROID_X', 'CENTROID_Y']].values
    vertiport_coords = vertiports_df[['X', 'Y']].values
    
    # Calculate distance matrix: zones vs vertiports
    print("📏 Calculating distances...")
    distances = cdist(zone_coords, vertiport_coords, metric='euclidean')
    
    # For each zone, find minimum distance to any vertiport
    min_distances = np.min(distances, axis=1)
    
    # Coverage criteria: within 5000m (5km)
    coverage_threshold = 5000  # meters
    zones_df['distance_to_nearest_vertiport'] = min_distances
    zones_df['is_covered'] = min_distances <= coverage_threshold
    
    # Calculate coverage statistics
    total_zones = len(zones_df)
    covered_zones = zones_df['is_covered'].sum()
    coverage_ratio_zones = covered_zones / total_zones * 100
    
    total_population = zones_df['Population'].sum()
    covered_population = zones_df[zones_df['is_covered']]['Population'].sum()
    coverage_ratio_population = covered_population / total_population * 100
    
    print(f"\n📈 COVERAGE RESULTS:")
    print(f"   Total zones: {total_zones:,}")
    print(f"   Covered zones: {covered_zones:,} ({coverage_ratio_zones:.1f}%)")
    print(f"   Not covered zones: {total_zones - covered_zones:,}")
    print(f"")
    print(f"   Total population: {total_population:,}")
    print(f"   Population in covered zones: {covered_population:,} ({coverage_ratio_population:.1f}%)")
    print(f"   Population in not covered zones: {total_population - covered_population:,}")
    
    # Save results
    output_file = "zone_coverage_results.csv"
    zones_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return zones_df

if __name__ == "__main__":
    results = calculate_zone_coverage()
