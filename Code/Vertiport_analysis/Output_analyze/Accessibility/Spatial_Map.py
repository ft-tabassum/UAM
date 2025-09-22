"""
Simple UAM Spatial Accessibility Map
===================================

Creates a simple spatial accessibility map showing the distribution 
of accessibility indices across job locations - just like the existing code style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_simple_spatial_map():
    """Create simple spatial accessibility map"""
    
    # Load results
    results_file = "/Result/Vertiport_analysis/Probability_clustering/Accessibility/uam_accessibility_results.csv"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run Accessibility.py first")
        return
        
    # Load data
    print("Loading accessibility results...")
    results_df = pd.read_csv(results_file)
    print(f"Loaded {len(results_df)} accessibility results")
    
    # Sample data for better visualization
    sample_size = min(15000, len(results_df))
    sample_df = results_df.sample(n=sample_size, random_state=42)
    
    # Create the map
    plt.figure(figsize=(12, 10))
    
    # Spatial Accessibility Map (like existing code)
    scatter = plt.scatter(sample_df['job_x'], sample_df['job_y'], 
                         c=sample_df['ptal_band'], 
                         cmap='viridis', s=2, alpha=0.6)
    
    plt.colorbar(scatter, label='PTAL Band')
    plt.xlabel('Job X Coordinate (meters)')
    plt.ylabel('Job Y Coordinate (meters)')
    plt.title('Spatial Distribution of UAM Accessibility')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the map
    output_dir = "/Result/Vertiport_analysis/Probability_clustering/PTAL_Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'simple_spatial_accessibility_map.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {output_file}")
    
    # Try to show the plot
    try:
        plt.show()
        print("Plot displayed!")
    except:
        print("Could not display plot window, but map was saved successfully!")
    
    print("Simple spatial accessibility map created!")

if __name__ == "__main__":
    create_simple_spatial_map()
