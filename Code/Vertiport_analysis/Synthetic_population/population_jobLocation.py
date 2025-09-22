"""
Data Cleaning and Merging Script for Population and Job Data

This script performs the following operations:
1. Removes -1 values from "personId" column in job data (jj_2011.csv)
2. Removes 0 values from "workplace" column in population data (pp_2011.csv)
3. Merges population and job datasets based on population "workplace" = job "id"

Author: Generated for UAM Thesis Project
Date: 2024
"""

import pandas as pd
import os
from pathlib import Path

def clean_and_merge_data():
    """
    Clean population and job data, then merge them based on workplace = id
    """
    
    # Define file paths
    job_file_path = "D:/Files_D/Study/==Thesis==/data/travel_demand_2021/travel_demand_2021/sp/jj_2011.csv"
    population_file_path = "D:/Files_D/Study/==Thesis==/data/travel_demand_2021/travel_demand_2021/sp/pp_2011.csv"
    
    # Output file path
    output_file_path = "D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/population_jobLocation.csv"
    
    print("Starting data cleaning and merging process...")
    
    try:
        # Step 1: Load and clean job data
        print("\n1. Loading job data...")
        job_df = pd.read_csv(job_file_path)
        print(f"   Original job data shape: {job_df.shape}")
        
        # Remove -1 values from personId column
        original_job_count = len(job_df)
        job_df_cleaned = job_df[job_df['personId'] != -1]
        removed_job_count = original_job_count - len(job_df_cleaned)
        print(f"   Removed {removed_job_count} rows with personId = -1")
        print(f"   Cleaned job data shape: {job_df_cleaned.shape}")
        
        # Step 2: Load and clean population data
        print("\n2. Loading population data...")
        population_df = pd.read_csv(population_file_path)
        print(f"   Original population data shape: {population_df.shape}")
        
        # Remove 0 values from workplace column
        original_pop_count = len(population_df)
        population_df_cleaned = population_df[population_df['workplace'] != 0]
        removed_pop_count = original_pop_count - len(population_df_cleaned)
        print(f"   Removed {removed_pop_count} rows with workplace = 0")
        
        # Remove unnecessary columns
        columns_to_drop = ['disability', 'schoolId']
        population_df_cleaned = population_df_cleaned.drop(columns=columns_to_drop)
        print(f"   Removed columns: {columns_to_drop}")
        print(f"   Cleaned population data shape: {population_df_cleaned.shape}")
        
        # Step 3: Merge datasets
        print("\n3. Merging datasets...")
        print(f"   Merging on: population['workplace'] = job['id']")
        
        # Perform the merge
        merged_df = pd.merge(
            population_df_cleaned, 
            job_df_cleaned, 
            left_on='workplace', 
            right_on='id', 
            how='inner'
        )
        
        print(f"   Merged data shape: {merged_df.shape}")
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'id_x': 'pp_id',           # Population person ID
            'id_y': 'jj_id',           # Job ID
            'zone': 'jobZone',         # Job zone
            'type': 'job_type',        # Job type
            'coordX': 'job_coordX',    # Job X coordinate
            'coordY': 'job_coordY',    # Job Y coordinate
            'startTime': 'job_startTime',  # Job start time
            'duration': 'job_duration'     # Job duration
        })
        
        print("   Column renaming completed:")
        print("   - id_x → pp_id (population person ID)")
        print("   - id_y → jj_id (job ID)")
        print("   - zone → jobZone (job zone)")
        print("   - type → job_type (job type)")
        print("   - coordX → job_coordX (job X coordinate)")
        print("   - coordY → job_coordY (job Y coordinate)")
        print("   - startTime → job_startTime (job start time)")
        print("   - duration → job_duration (job duration)")
        
        # Step 4: Display merge statistics
        print("\n4. Merge Statistics:")
        print(f"   Population records before merge: {len(population_df_cleaned)}")
        print(f"   Job records before merge: {len(job_df_cleaned)}")
        print(f"   Successful matches: {len(merged_df)}")
        print(f"   Population records not matched: {len(population_df_cleaned) - len(merged_df)}")
        print(f"   Job records not matched: {len(job_df_cleaned) - len(merged_df)}")
        
        # Step 5: Display column information
        print("\n5. Column Information:")
        print(f"   Population columns: {list(population_df_cleaned.columns)}")
        print(f"   Job columns: {list(job_df_cleaned.columns)}")
        print(f"   Merged columns: {list(merged_df.columns)}")
        
        # Step 6: Save merged data
        print(f"\n6. Saving merged data to: {output_file_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        merged_df.to_csv(output_file_path, index=False)
        print(f"   Successfully saved {len(merged_df)} records")
        
        # Step 7: Display sample of merged data
        print("\n7. Sample of merged data:")
        print(merged_df.head())
        
        # Step 8: Display data types
        print("\n8. Data types:")
        print(merged_df.dtypes)
        
        print("\n✓ Data cleaning and merging completed successfully!")
        
        return merged_df
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please check the file paths and ensure the files exist.")
        return None
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def display_data_summary(df, dataset_name):
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
    # Run the data cleaning and merging process
    merged_data = clean_and_merge_data()
    
    if merged_data is not None:
        print(f"\nFinal merged dataset contains {len(merged_data)} records")
        print("Script execution completed successfully!")
    else:
        print("Script execution failed. Please check the error messages above.")
