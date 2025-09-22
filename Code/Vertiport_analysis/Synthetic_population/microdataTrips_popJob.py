"""
Trip and Population-Job Data Merging Script

This script merges trip data with population-job data based on:
trip['person_id'] = popJob['pp_id']

Author: Generated for UAM Thesis Project
Date: 2024
"""

import pandas as pd
import os
from pathlib import Path

def merge_trips_with_population_jobs():
    """
    Merge trip data with population-job data based on person_id = pp_id
    """
    
    # Define file paths
    trip_file_path = "D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/microdata_trips_purpose.csv"
    popjob_file_path = "D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/population_jobLocation.csv"
    
    # Output file path
    output_file_path = "D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/microdataTrips_popJob.csv"
    
    print("Starting trip and population-job data merging process...")
    
    try:
        # Step 1: Load trip data
        print("\n1. Loading trip data...")
        trip_df = pd.read_csv(trip_file_path)
        print(f"   Trip data shape: {trip_df.shape}")
        print(f"   Trip columns: {list(trip_df.columns)}")
        
        # Step 2: Load population-job data
        print("\n2. Loading population-job data...")
        popjob_df = pd.read_csv(popjob_file_path)
        print(f"   Population-job data shape: {popjob_df.shape}")
        print(f"   Population-job columns: {list(popjob_df.columns)}")
        
        # Step 3: Check merge keys
        print("\n3. Checking merge keys...")
        print(f"   Trip 'person_id' column exists: {'person_id' in trip_df.columns}")
        print(f"   PopJob 'pp_id' column exists: {'pp_id' in popjob_df.columns}")
        
        if 'person_id' in trip_df.columns:
            print(f"   Unique person_id values in trip data: {trip_df['person_id'].nunique()}")
        if 'pp_id' in popjob_df.columns:
            print(f"   Unique pp_id values in popjob data: {popjob_df['pp_id'].nunique()}")
        
        # Step 4: Perform the merge
        print("\n4. Merging datasets...")
        print(f"   Merging on: trip['person_id'] = popjob['pp_id']")
        
        # Perform the merge
        merged_df = pd.merge(
            trip_df, 
            popjob_df, 
            left_on='person_id', 
            right_on='pp_id', 
            how='inner'
        )
        
        print(f"   Merged data shape: {merged_df.shape}")
        
        # Step 5: Clean duplicate columns and unnecessary columns
        print("\n5. Cleaning duplicate and unnecessary columns...")
        
        # Remove duplicate columns (keep only one version of each)
        columns_to_remove = [
            'age_x', 'gender_x', 'occupation_x', 'driversLicense_x', 'income_x',
            'age_y', 'gender_y', 'occupation_y', 'driversLicense_y',
            'departure_time_return', 'pp_id', 'hhid', 'jj_id', 'personId'
        ]
        
        # Check which columns actually exist in the merged data
        existing_columns_to_remove = [col for col in columns_to_remove if col in merged_df.columns]
        non_existing_columns = [col for col in columns_to_remove if col not in merged_df.columns]
        
        if existing_columns_to_remove:
            merged_df = merged_df.drop(columns=existing_columns_to_remove)
            print(f"   Removed columns: {existing_columns_to_remove}")
        
        if non_existing_columns:
            print(f"   Columns not found (may have different names): {non_existing_columns}")
        
        print(f"   Cleaned merged data shape: {merged_df.shape}")
        
        # Step 6: Display merge statistics
        print("\n6. Merge Statistics:")
        print(f"   Trip records before merge: {len(trip_df)}")
        print(f"   PopJob records before merge: {len(popjob_df)}")
        print(f"   Successful matches: {len(merged_df)}")
        print(f"   Trip records not matched: {len(trip_df) - len(merged_df)}")
        print(f"   PopJob records not matched: {len(popjob_df) - len(merged_df)}")
        
        # Calculate match rates
        trip_match_rate = len(merged_df) / len(trip_df) * 100
        popjob_match_rate = len(merged_df) / len(popjob_df) * 100
        print(f"   Trip match rate: {trip_match_rate:.1f}%")
        print(f"   PopJob match rate: {popjob_match_rate:.1f}%")
        
        # Step 7: Display column information
        print("\n7. Column Information:")
        print(f"   Trip columns: {list(trip_df.columns)}")
        print(f"   PopJob columns: {list(popjob_df.columns)}")
        print(f"   Final merged columns: {list(merged_df.columns)}")
        print(f"   Total columns in final merged data: {len(merged_df.columns)}")
        
        # Step 8: Check for remaining duplicate columns (if any)
        duplicate_columns = [col for col in merged_df.columns if col.endswith('_x') or col.endswith('_y')]
        if duplicate_columns:
            print(f"   Remaining duplicate columns found: {duplicate_columns}")
        else:
            print("   No remaining duplicate columns found")
        
        # Step 9: Save merged data
        print(f"\n8. Saving merged data to: {output_file_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        merged_df.to_csv(output_file_path, index=False)
        print(f"   Successfully saved {len(merged_df)} records")
        
        # Step 10: Display sample of merged data
        print("\n9. Sample of merged data:")
        print(merged_df.head())
        
        # Step 11: Display data types
        print("\n10. Data types:")
        print(merged_df.dtypes)
        
        # Step 12: Display file size information
        file_size = os.path.getsize(output_file_path) / (1024 * 1024)  # Size in MB
        print(f"\n11. Output file size: {file_size:.1f} MB")
        
        print("\n✓ Trip and population-job data merging completed successfully!")
        
        return merged_df
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please check the file paths and ensure the files exist.")
        return None
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_merge_summary(df, dataset_name):
    """
    Display summary statistics for a dataset
    """
    if df is not None:
        print(f"\n{dataset_name} Summary:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)

if __name__ == "__main__":
    # Run the trip and population-job data merging process
    merged_data = merge_trips_with_population_jobs()
    
    if merged_data is not None:
        print(f"\nFinal merged dataset contains {len(merged_data)} records")
        print("Script execution completed successfully!")
    else:
        print("Script execution failed. Please check the error messages above.")
