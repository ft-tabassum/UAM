"""
Script to merge LightGBM predictions with synthetic population trip data
Merges based on trip_id and keeps specified columns from each dataset
"""

import pandas as pd
import os
from pathlib import Path

def merge_synthetic_predictions():
    """
    Merge LightGBM synthetic population predictions with trip data
    """
    
    # Define file paths using the base path [[memory:6463058]]
    base_path = "//"
    lightgbm_file = os.path.join(base_path, "Result/Vertiport_analysis/Probability_clustering/Weighting/5km_radius_LightGBM_synthetic_population_predictions_weights.csv")
    trips_file = os.path.join(base_path, "Result/Vertiport_analysis/Synthetic_population/microdataTrips_popJob.csv")
    output_file = os.path.join(base_path, "Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_weightPredict_job.csv")
    
    print("Starting merge process...")
    print(f"LightGBM file: {lightgbm_file}")
    print(f"Trips file: {trips_file}")
    print(f"Output file: {output_file}")
    
    # Check if files exist
    if not os.path.exists(lightgbm_file):
        raise FileNotFoundError(f"LightGBM file not found: {lightgbm_file}")
    
    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"Trips file not found: {trips_file}")
    
    try:
        # Read LightGBM predictions file (keep all columns)
        print("\nReading LightGBM predictions file...")
        df_lightgbm = pd.read_csv(lightgbm_file)
        print(f"LightGBM file shape: {df_lightgbm.shape}")
        print(f"LightGBM columns: {list(df_lightgbm.columns)}")
        
        # Read trips file and select only required columns
        print("\nReading trips file...")
        columns_to_keep = ["trip_id", "purpose", "workplace", "homeZone", "jobZone", 
                          "job_type", "job_coordX", "job_coordY", "job_startTime"]
        
        df_trips = pd.read_csv(trips_file, usecols=columns_to_keep)
        print(f"Trips file shape: {df_trips.shape}")
        print(f"Selected columns: {list(df_trips.columns)}")
        
        # Check if trip_id exists in both files
        if 'trip_id' not in df_lightgbm.columns:
            raise ValueError("trip_id column not found in LightGBM file")
        if 'trip_id' not in df_trips.columns:
            raise ValueError("trip_id column not found in trips file")
        
        # Display basic statistics about trip_ids
        print(f"\nTrip ID statistics:")
        print(f"LightGBM unique trip_ids: {df_lightgbm['trip_id'].nunique()}")
        print(f"Trips unique trip_ids: {df_trips['trip_id'].nunique()}")
        print(f"Common trip_ids: {len(set(df_lightgbm['trip_id']) & set(df_trips['trip_id']))}")
        
        # Perform left join to keep all records from LightGBM file
        print("\nPerforming merge...")
        df_merged = pd.merge(df_lightgbm, df_trips, on='trip_id', how='left')
        print(f"Merged file shape: {df_merged.shape}")
        
        # Check for missing values in merged columns
        missing_counts = df_merged[["purpose", "workplace", "homeZone", "jobZone", 
                                  "job_type", "job_coordX", "job_coordY", "job_startTime"]].isnull().sum()
        print(f"\nMissing values in merged columns:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df_merged)*100:.2f}%)")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save merged file
        print(f"\nSaving merged file to: {output_file}")
        df_merged.to_csv(output_file, index=False)
        print("Merge completed successfully!")
        
        # Display final summary
        print(f"\nFinal merged dataset:")
        print(f"  Shape: {df_merged.shape}")
        print(f"  Columns: {len(df_merged.columns)}")
        print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return df_merged
        
    except Exception as e:
        print(f"Error during merge process: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("=== Starting merge process ===")
        # Run the merge
        merged_data = merge_synthetic_predictions()
        
        # Display first few rows of merged data
        print("\nFirst 5 rows of merged data:")
        print(merged_data.head())
        
        print("\nColumn names in merged data:")
        for i, col in enumerate(merged_data.columns, 1):
            print(f"{i:2d}. {col}")
            
        print("\n=== Merge process completed successfully ===")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
