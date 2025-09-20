"""
UAM Accessibility Index (UAI) Calculation Script
================================================

This script implements the UAM Accessibility Index based on the methodology from 
public transport accessibility studies, adapted for Urban Air Mobility (UAM) systems.

The UAI measures how accessible different areas are to UAM services by calculating:
1. Total Access Time (TAT) = Walk/Access Time + Pre-flight Time + Airborne Time
2. Equivalent Doorstep Frequency (EDF) = 30 / TAT
3. Accessibility Index weighted by UAM mode choice probabilities
4. Accessibility bands (similar to PTAL - Public Transport Accessibility Levels)

Author: AI Assistant
Date: 2024
Based on: Weighted_clustering.py and accessibility index methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND PARAMETERS
# =============================================================================

class UAIConfig:
    """Configuration class for UAM Accessibility Index calculation"""
    
    # File paths
    DATA_PATH = "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv"
    VERTIPORT_PATH = "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_final.csv"
    OUTPUT_DIR = "D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Accessibility_Index"
    
    # UAM system parameters (from Weighted_clustering.py)
    PRE_FLIGHT_TIME = 15  # minutes - equivalent to waiting time in PT
    UAM_CRUISE_SPEED = 4166.67  # m/min (250 km/h)
    WALKING_SPEED = 83.33  # m/min (5 km/h)
    CAR_SPEED = 418.33  # m/min (25.1 km/h)
    CIRCUITY_FACTOR = 1.215  # for car routing
    
    # Catchment areas
    WALKING_CATCHMENT = 1000  # meters (1 km)
    CAR_CATCHMENT = 5000  # meters (5 km)
    
    # Accessibility bands (similar to PTAL 1-8)
    ACCESSIBILITY_BANDS = {
        'Excellent': (0.8, 1.0),      # Very high accessibility
        'Very Good': (0.6, 0.8),      # High accessibility  
        'Good': (0.4, 0.6),           # Moderate-high accessibility
        'Fair': (0.3, 0.4),           # Moderate accessibility
        'Below Average': (0.2, 0.3),  # Moderate-low accessibility
        'Poor': (0.1, 0.2),           # Low accessibility
        'Very Poor': (0.05, 0.1),     # Very low accessibility
        'Inaccessible': (0.0, 0.05)   # Minimal/no accessibility
    }

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """
    Load the synthetic population data with UAM predictions and vertiport coordinates
    
    Returns:
        tuple: (population_data, vertiport_coords)
    """
    logger.info("Loading synthetic population data...")
    
    # Load population data ###################check population data
    population_data = pd.read_csv(UAIConfig.DATA_PATH, low_memory=False)
    logger.info(f"Loaded {len(population_data)} trips from synthetic population")
    
    # Load vertiport coordinates ###################check vertiport coordinates
    vertiport_coords = pd.read_csv(UAIConfig.VERTIPORT_PATH)
    vertiport_coords_array = vertiport_coords[['X', 'Y']].values
    logger.info(f"Loaded {len(vertiport_coords_array)} vertiports")
    
    return population_data, vertiport_coords_array

def validate_data(population_data, vertiport_coords):
    """
    Validate that required columns exist in the data
    
    Args:
        population_data (pd.DataFrame): Population data
        vertiport_coords (np.array): Vertiport coordinates
    """
    required_columns = [
        'originX', 'originY', 'destinationX', 'destinationY',
        'prob_mode_Autonomous Flying Taxi', 'travel_time_Uam',
        'uam_origin_vertiport', 'uam_dest_vertiport'
    ]
    
    missing_columns = [col for col in required_columns if col not in population_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Data validation passed - all required columns present")

# =============================================================================
# ACCESSIBILITY CALCULATION FUNCTIONS
# =============================================================================

def calculate_access_time_to_vertiport(origin_coords, vertiport_coords, access_mode='walk'):
    """
    Calculate access time from origin to nearest vertiport
    
    Args:
        origin_coords (np.array): Origin coordinates (N, 2)
        vertiport_coords (np.array): All vertiport coordinates (M, 2)
        access_mode (str): 'walk' or 'car'
    
    Returns:
        tuple: (access_times, nearest_vertiport_indices, distances)
    """
    # Calculate distances to all vertiports
    distances = cdist(origin_coords, vertiport_coords)
    
    # Find nearest vertiport for each origin
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(origin_coords)), nearest_indices]
    
    # Apply circuity factor for car access
    if access_mode == 'car':
        effective_distances = nearest_distances * UAIConfig.CIRCUITY_FACTOR
        speed = UAIConfig.CAR_SPEED
    else:  # walking
        effective_distances = nearest_distances
        speed = UAIConfig.WALKING_SPEED
    
    # Calculate access times in minutes
    access_times = effective_distances / speed
    
    return access_times, nearest_indices, nearest_distances

def calculate_uam_airborne_time(origin_vertiport_idx, dest_vertiport_idx, vertiport_coords):
    """
    Calculate airborne time between vertiports
    
    Args:
        origin_vertiport_idx (np.array): Origin vertiport indices
        dest_vertiport_idx (np.array): Destination vertiport indices  
        vertiport_coords (np.array): Vertiport coordinates
    
    Returns:
        np.array: Airborne times in minutes
    """
    # Get vertiport coordinates for each trip
    origin_vertiports = vertiport_coords[origin_vertiport_idx]
    dest_vertiports = vertiport_coords[dest_vertiport_idx]
    
    # Calculate Euclidean distances between vertiports
    airborne_distances = np.linalg.norm(origin_vertiports - dest_vertiports, axis=1)
    
    # Calculate airborne times
    airborne_times = airborne_distances / UAIConfig.UAM_CRUISE_SPEED
    
    return airborne_times

def calculate_equivalent_doorstep_frequency(total_access_time):
    """
    Calculate Equivalent Doorstep Frequency (EDF) based on total access time
    
    EDF = 10 / TAT (where TAT is in minutes) source: Peksa, M., & Bogenberger, K. (2020)
    This converts access time into an equivalent service frequency
    Using 10 instead of 30 as UAM services have different frequency characteristics
    
    Args:
        total_access_time (np.array): Total access time in minutes
    
    Returns:
        np.array: Equivalent Doorstep Frequency
    """
    # Avoid division by zero
    total_access_time = np.maximum(total_access_time, 0.1)
    edf = 10.0 / total_access_time
    return edf

def calculate_accessibility_index(edf_origin, edf_destination, uam_probabilities):
    """
    Calculate weighted Accessibility Index for UAM
    
    Args:
        edf_origin (np.array): EDF for origin access
        edf_destination (np.array): EDF for destination access  
        uam_probabilities (np.array): UAM mode choice probabilities
    
    Returns:
        np.array: Accessibility Index values
    """
    # Use the minimum EDF (worst access) as the limiting factor
    # This ensures accessibility reflects the most constrained access
    min_edf = np.minimum(edf_origin, edf_destination)
    
    # Weight by UAM probability - higher probability = higher accessibility
    accessibility_index = min_edf * uam_probabilities
    
    return accessibility_index

def assign_accessibility_bands(accessibility_index):
    """
    Assign accessibility bands based on normalized accessibility index
    
    Args:
        accessibility_index (np.array): Raw accessibility index values
    
    Returns:
        tuple: (band_labels, normalized_scores)
    """
    # Normalize accessibility index to 0-1 range
    max_val = np.max(accessibility_index)
    min_val = np.min(accessibility_index)
    
    if max_val > min_val:
        normalized_scores = (accessibility_index - min_val) / (max_val - min_val)
    else:
        normalized_scores = np.ones_like(accessibility_index) * 0.5
    
    # Assign bands
    band_labels = np.full(len(accessibility_index), 'Inaccessible', dtype=object)
    
    for band_name, (min_threshold, max_threshold) in UAIConfig.ACCESSIBILITY_BANDS.items():
        mask = (normalized_scores >= min_threshold) & (normalized_scores < max_threshold)
        band_labels[mask] = band_name
    
    # Handle edge case for maximum values
    band_labels[normalized_scores >= 1.0] = 'Excellent'
    
    return band_labels, normalized_scores

# =============================================================================
# MAIN UAI CALCULATION FUNCTION
# =============================================================================

def calculate_uam_accessibility_index(population_data, vertiport_coords):
    """
    Main function to calculate UAM Accessibility Index for all trips
    
    Args:
        population_data (pd.DataFrame): Trip data with UAM predictions
        vertiport_coords (np.array): Vertiport coordinates
    
    Returns:
        pd.DataFrame: Population data with UAI calculations added
    """
    logger.info("Calculating UAM Accessibility Index...")
    
    # Extract coordinates
    origins = population_data[['originX', 'originY']].values
    destinations = population_data[['destinationX', 'destinationY']].values
    
    # Get UAM probabilities and vertiport assignments
    uam_probs = population_data['prob_mode_Autonomous Flying Taxi'].values
    origin_vertiport_idx = population_data['uam_origin_vertiport'].values
    dest_vertiport_idx = population_data['uam_dest_vertiport'].values
    
    logger.info("Calculating access times...")
    
    # Calculate access times to vertiports
    origin_access_times, _, origin_distances = calculate_access_time_to_vertiport(
        origins, vertiport_coords, access_mode='walk'
    )
    dest_access_times, _, dest_distances = calculate_access_time_to_vertiport(
        destinations, vertiport_coords, access_mode='walk'
    )
    
    logger.info("Calculating airborne times...")
    
    # Calculate airborne times
    airborne_times = calculate_uam_airborne_time(
        origin_vertiport_idx, dest_vertiport_idx, vertiport_coords
    )
    
    logger.info("Calculating total access times...")
    
    # Calculate Total Access Time (TAT)
    total_access_time = (UAIConfig.PRE_FLIGHT_TIME + 
                        origin_access_times + 
                        airborne_times + 
                        dest_access_times)
    
    logger.info("Calculating Equivalent Doorstep Frequency...")
    
    # Calculate EDF for origin and destination access
    edf_origin = calculate_equivalent_doorstep_frequency(
        UAIConfig.PRE_FLIGHT_TIME + origin_access_times
    )
    edf_destination = calculate_equivalent_doorstep_frequency(
        UAIConfig.PRE_FLIGHT_TIME + dest_access_times
    )
    
    logger.info("Calculating Accessibility Index...")
    
    # Calculate weighted Accessibility Index
    accessibility_index = calculate_accessibility_index(
        edf_origin, edf_destination, uam_probs
    )
    
    logger.info("Assigning accessibility bands...")
    
    # Assign accessibility bands
    band_labels, normalized_scores = assign_accessibility_bands(accessibility_index)
    
    # Create output dataframe with all calculations
    output_data = population_data.copy()
    
    # Add UAI calculations
    output_data['origin_access_time'] = origin_access_times
    output_data['dest_access_time'] = dest_access_times
    output_data['airborne_time'] = airborne_times
    output_data['total_access_time'] = total_access_time
    output_data['origin_access_distance'] = origin_distances
    output_data['dest_access_distance'] = dest_distances
    output_data['edf_origin'] = edf_origin
    output_data['edf_destination'] = edf_destination
    output_data['accessibility_index'] = accessibility_index
    output_data['accessibility_band'] = band_labels
    output_data['accessibility_score_normalized'] = normalized_scores
    
    logger.info(f"UAI calculation complete. Added {len(output_data.columns) - len(population_data.columns)} new columns")
    
    return output_data

# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================

def create_output_directory():
    """Create output directory for results"""
    os.makedirs(UAIConfig.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory created: {UAIConfig.OUTPUT_DIR}")

def generate_accessibility_statistics(output_data):
    """
    Generate comprehensive accessibility statistics
    
    Args:
        output_data (pd.DataFrame): Data with UAI calculations
    
    Returns:
        dict: Statistics dictionary
    """
    logger.info("Generating accessibility statistics...")
    
    stats = {}
    
    # Basic statistics
    stats['total_trips'] = len(output_data)
    stats['mean_accessibility_index'] = output_data['accessibility_index'].mean()
    stats['std_accessibility_index'] = output_data['accessibility_index'].std()
    stats['min_accessibility_index'] = output_data['accessibility_index'].min()
    stats['max_accessibility_index'] = output_data['accessibility_index'].max()
    
    # Time statistics
    stats['mean_total_access_time'] = output_data['total_access_time'].mean()
    stats['mean_origin_access_time'] = output_data['origin_access_time'].mean()
    stats['mean_dest_access_time'] = output_data['dest_access_time'].mean()
    stats['mean_airborne_time'] = output_data['airborne_time'].mean()
    
    # Distance statistics
    stats['mean_origin_access_distance'] = output_data['origin_access_distance'].mean()
    stats['mean_dest_access_distance'] = output_data['dest_access_distance'].mean()
    
    # EDF statistics
    stats['mean_edf_origin'] = output_data['edf_origin'].mean()
    stats['mean_edf_destination'] = output_data['edf_destination'].mean()
    
    # Band distribution
    band_counts = output_data['accessibility_band'].value_counts()
    band_percentages = output_data['accessibility_band'].value_counts(normalize=True) * 100
    stats['band_distribution'] = band_counts
    stats['band_percentages'] = band_percentages
    
    return stats

def plot_accessibility_distribution(output_data, stats):
    """
    Create comprehensive accessibility visualization plots
    
    Args:
        output_data (pd.DataFrame): Data with UAI calculations
        stats (dict): Accessibility statistics
    """
    logger.info("Creating accessibility visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accessibility Index Distribution
    plt.subplot(3, 3, 1)
    plt.hist(output_data['accessibility_index'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(stats['mean_accessibility_index'], color='red', linestyle='--', 
                label=f"Mean: {stats['mean_accessibility_index']:.4f}")
    plt.xlabel('Accessibility Index')
    plt.ylabel('Frequency')
    plt.title('Distribution of UAM Accessibility Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accessibility Bands Distribution
    plt.subplot(3, 3, 2)
    band_counts = stats['band_distribution']
    colors = plt.cm.Set3(np.linspace(0, 1, len(band_counts)))
    plt.pie(band_counts.values, labels=band_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Distribution of Accessibility Bands')
    
    # 3. Total Access Time Distribution
    plt.subplot(3, 3, 3)
    plt.hist(output_data['total_access_time'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(stats['mean_total_access_time'], color='red', linestyle='--',
                label=f"Mean: {stats['mean_total_access_time']:.1f} min")
    plt.xlabel('Total Access Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Access Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Access Time Components
    plt.subplot(3, 3, 4)
    time_components = ['Origin Access', 'Airborne', 'Destination Access']
    time_means = [stats['mean_origin_access_time'], stats['mean_airborne_time'], stats['mean_dest_access_time']]
    plt.bar(time_components, time_means, alpha=0.7)
    plt.ylabel('Time (minutes)')
    plt.title('Average Access Time Components')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Accessibility vs UAM Probability
    plt.subplot(3, 3, 5)
    plt.scatter(output_data['prob_mode_Autonomous Flying Taxi'], output_data['accessibility_index'], 
                alpha=0.5, s=1)
    plt.xlabel('UAM Probability')
    plt.ylabel('Accessibility Index')
    plt.title('Accessibility Index vs UAM Probability')
    plt.grid(True, alpha=0.3)
    
    # 6. EDF Distribution
    plt.subplot(3, 3, 6)
    plt.hist(output_data['edf_origin'], bins=50, alpha=0.7, label='Origin EDF', edgecolor='black')
    plt.hist(output_data['edf_destination'], bins=50, alpha=0.7, label='Destination EDF', edgecolor='black')
    plt.xlabel('Equivalent Doorstep Frequency')
    plt.ylabel('Frequency')
    plt.title('Distribution of EDF (Origin vs Destination)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Access Distance Distribution
    plt.subplot(3, 3, 7)
    plt.hist(output_data['origin_access_distance']/1000, bins=50, alpha=0.7, label='Origin', edgecolor='black')
    plt.hist(output_data['dest_access_distance']/1000, bins=50, alpha=0.7, label='Destination', edgecolor='black')
    plt.xlabel('Access Distance (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Access Distances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Accessibility Band vs Trip Length
    plt.subplot(3, 3, 8)
    band_order = list(UAIConfig.ACCESSIBILITY_BANDS.keys())
    band_data = []
    for band in band_order:
        band_trips = output_data[output_data['accessibility_band'] == band]['trip_length']
        band_data.append(band_trips)
    
    plt.boxplot(band_data, labels=band_order)
    plt.ylabel('Trip Length (meters)')
    plt.title('Trip Length by Accessibility Band')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 9. Spatial Accessibility (if coordinates are available)
    plt.subplot(3, 3, 9)
    scatter = plt.scatter(output_data['originX'], output_data['originY'], 
                         c=output_data['accessibility_index'], 
                         cmap='viridis', s=0.5, alpha=0.6)
    plt.colorbar(scatter, label='Accessibility Index')
    plt.xlabel('Origin X Coordinate (meters)')
    plt.ylabel('Origin Y Coordinate (meters)')
    plt.title('Spatial Distribution of Accessibility')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(UAIConfig.OUTPUT_DIR, 'accessibility_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Accessibility visualization saved: {plot_path}")

def save_results(output_data, stats):
    """
    Save all results to files
    
    Args:
        output_data (pd.DataFrame): Data with UAI calculations
        stats (dict): Accessibility statistics
    """
    logger.info("Saving results...")
    
    # Save main results
    output_path = os.path.join(UAIConfig.OUTPUT_DIR, 'uam_accessibility_results.csv')
    output_data.to_csv(output_path, index=False)
    logger.info(f"Main results saved: {output_path}")
    
    # Save statistics
    stats_path = os.path.join(UAIConfig.OUTPUT_DIR, 'accessibility_statistics.csv')
    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ['Value']
    stats_df.to_csv(stats_path)
    logger.info(f"Statistics saved: {stats_path}")
    
    # Save band distribution
    band_dist_path = os.path.join(UAIConfig.OUTPUT_DIR, 'accessibility_band_distribution.csv')
    band_df = pd.DataFrame({
        'Band': stats['band_distribution'].index,
        'Count': stats['band_distribution'].values,
        'Percentage': stats['band_percentages'].values
    })
    band_df.to_csv(band_dist_path, index=False)
    logger.info(f"Band distribution saved: {band_dist_path}")

def generate_summary_report(stats):
    """
    Generate a comprehensive summary report
    
    Args:
        stats (dict): Accessibility statistics
    """
    logger.info("Generating summary report...")
    
    report_path = os.path.join(UAIConfig.OUTPUT_DIR, 'accessibility_summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("UAM ACCESSIBILITY INDEX (UAI) ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("METHODOLOGY OVERVIEW:\n")
        f.write("-" * 30 + "\n")
        f.write("The UAM Accessibility Index (UAI) measures how accessible different areas are\n")
        f.write("to UAM services, adapted from public transport accessibility methodologies.\n\n")
        
        f.write("Key Components:\n")
        f.write("1. Total Access Time (TAT) = Pre-flight + Access + Airborne + Access\n")
        f.write("2. Equivalent Doorstep Frequency (EDF) = 10 / TAT\n")
        f.write("3. Accessibility Index = min(EDF_origin, EDF_dest) × UAM_probability\n")
        f.write("4. Accessibility Bands: 8 levels from 'Excellent' to 'Inaccessible'\n\n")
        
        f.write("SYSTEM PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Pre-flight Time: {UAIConfig.PRE_FLIGHT_TIME} minutes\n")
        f.write(f"UAM Cruise Speed: {UAIConfig.UAM_CRUISE_SPEED:.1f} m/min ({UAIConfig.UAM_CRUISE_SPEED*60/1000:.0f} km/h)\n")
        f.write(f"Walking Speed: {UAIConfig.WALKING_SPEED:.1f} m/min ({UAIConfig.WALKING_SPEED*60/1000:.0f} km/h)\n")
        f.write(f"Walking Catchment: {UAIConfig.WALKING_CATCHMENT/1000:.1f} km\n")
        f.write(f"Car Catchment: {UAIConfig.CAR_CATCHMENT/1000:.1f} km\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Trips Analyzed: {stats['total_trips']:,}\n")
        f.write(f"Mean Accessibility Index: {stats['mean_accessibility_index']:.4f}\n")
        f.write(f"Std Accessibility Index: {stats['std_accessibility_index']:.4f}\n")
        f.write(f"Min Accessibility Index: {stats['min_accessibility_index']:.4f}\n")
        f.write(f"Max Accessibility Index: {stats['max_accessibility_index']:.4f}\n\n")
        
        f.write("ACCESS TIME STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Total Access Time: {stats['mean_total_access_time']:.1f} minutes\n")
        f.write(f"Mean Origin Access Time: {stats['mean_origin_access_time']:.1f} minutes\n")
        f.write(f"Mean Destination Access Time: {stats['mean_dest_access_time']:.1f} minutes\n")
        f.write(f"Mean Airborne Time: {stats['mean_airborne_time']:.1f} minutes\n\n")
        
        f.write("ACCESS DISTANCE STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Origin Access Distance: {stats['mean_origin_access_distance']:.0f} meters ({stats['mean_origin_access_distance']/1000:.1f} km)\n")
        f.write(f"Mean Destination Access Distance: {stats['mean_dest_access_distance']:.0f} meters ({stats['mean_dest_access_distance']/1000:.1f} km)\n\n")
        
        f.write("ACCESSIBILITY BAND DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for band, count in stats['band_distribution'].items():
            percentage = stats['band_percentages'][band]
            f.write(f"{band:15}: {count:6,} trips ({percentage:5.1f}%)\n")
        
        f.write(f"\nAll results saved to: {UAIConfig.OUTPUT_DIR}\n")
        f.write("Files generated:\n")
        f.write("- uam_accessibility_results.csv (main results)\n")
        f.write("- accessibility_statistics.csv (summary statistics)\n")
        f.write("- accessibility_band_distribution.csv (band distribution)\n")
        f.write("- accessibility_analysis.png (visualizations)\n")
        f.write("- accessibility_summary_report.txt (this report)\n")
    
    logger.info(f"Summary report saved: {report_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    logger.info("Starting UAM Accessibility Index calculation...")
    
    try:
        # Create output directory
        create_output_directory()
        
        # Load and validate data
        population_data, vertiport_coords = load_data()
        validate_data(population_data, vertiport_coords)
        
        # Calculate UAM Accessibility Index
        output_data = calculate_uam_accessibility_index(population_data, vertiport_coords)
        
        # Generate statistics
        stats = generate_accessibility_statistics(output_data)
        
        # Create visualizations
        plot_accessibility_distribution(output_data, stats)
        
        # Save results
        save_results(output_data, stats)
        
        # Generate summary report
        generate_summary_report(stats)
        
        logger.info("UAM Accessibility Index calculation completed successfully!")
        logger.info(f"Results saved to: {UAIConfig.OUTPUT_DIR}")
        
        # Print key statistics to console
        print("\n" + "="*60)
        print("UAM ACCESSIBILITY INDEX - KEY RESULTS")
        print("="*60)
        print(f"Total Trips: {stats['total_trips']:,}")
        print(f"Mean Accessibility Index: {stats['mean_accessibility_index']:.4f}")
        print(f"Mean Total Access Time: {stats['mean_total_access_time']:.1f} minutes")
        print(f"\nAccessibility Band Distribution:")
        for band, count in stats['band_distribution'].items():
            percentage = stats['band_percentages'][band]
            print(f"  {band:15}: {percentage:5.1f}%")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
