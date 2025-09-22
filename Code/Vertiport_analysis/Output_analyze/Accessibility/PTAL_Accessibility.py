"""
UAM PTAL-Style Accessibility Calculation
=======================================

This script calculates accessibility indices for UAM (Urban Air Mobility) 
using a PTAL (Public Transport Accessibility Level) methodology adapted for vertiports.

Formula:
- WT = dest_access_time (from data)
- AWT = (4/2) + 1.5 = 3.5 minutes (UAM specific)
- TAT = WT + AWT
- EDF = 10/TAT
- AI = Sum of EDF values (weighted)
- PTAL = Classification into bands 1-10

Data Sources:
- POI (Job coordinates): merged_synthetic_predictions.csv
- SAP (Vertiport coordinates): optimized_vertiport_coords_final.csv
- Access times: dest_access_time column in merged_synthetic_predictions.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class UAMAccessibilityCalculator:
    def __init__(self, base_path="D:/Thesis/UAM/"):
        self.base_path = base_path
        self.awt_constant = 3.5  # AWT = (4/2) + 1.5 = 3.5 minutes for UAM
        
        # File paths
        self.jobs_file = os.path.join(base_path, "Result/Vertiport_analysis/Synthetic_population/merged_synthetic_predictions.csv")
        self.vertiports_file = os.path.join(base_path, "Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_final.csv")
        self.output_dir = os.path.join(base_path, "Result/Vertiport_analysis/Probability_clustering/Accessibility")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load job and vertiport data"""
        print("Loading data...")
        
        # Load job data
        print(f"Loading job data from: {self.jobs_file}")
        self.jobs_df = pd.read_csv(self.jobs_file)
        print(f"Loaded {len(self.jobs_df)} job records")
        
        # Load vertiport data
        print(f"Loading vertiport data from: {self.vertiports_file}")
        self.vertiports_df = pd.read_csv(self.vertiports_file)
        print(f"Loaded {len(self.vertiports_df)} vertiport records")
        
        # Rename vertiport columns to match expected format
        if 'X' in self.vertiports_df.columns and 'Y' in self.vertiports_df.columns:
            self.vertiports_df = self.vertiports_df.rename(columns={'X': 'x', 'Y': 'y'})
        
        # Display column information
        print("\nJob data columns:", self.jobs_df.columns.tolist())
        print("Vertiport data columns:", self.vertiports_df.columns.tolist())
        
    def calculate_accessibility(self):
        """Calculate accessibility indices for each job location"""
        print("\nCalculating accessibility indices...")
        
        # Check if dest_access_time column exists
        if 'dest_access_time' not in self.jobs_df.columns:
            print("Error: 'dest_access_time' column not found in job data")
            print("Available columns:", self.jobs_df.columns.tolist())
            return None
            
        # Calculate accessibility for each job
        accessibility_results = []
        
        for idx, job in self.jobs_df.iterrows():
            if idx % 10000 == 0:
                print(f"Processing job {idx}/{len(self.jobs_df)}")
                
            # Get job coordinates and access time
            job_x = job.get('job_coordX', None)
            job_y = job.get('job_coordY', None)
            dest_access_time = job.get('dest_access_time', None)
            
            if pd.isna(job_x) or pd.isna(job_y) or pd.isna(dest_access_time):
                continue
                
            # Calculate TAT and EDF
            wt = dest_access_time  # Walk/Drive Time
            awt = self.awt_constant  # Average Waiting Time (constant for UAM)
            tat = wt + awt  # Total Access Time
            edf = 10 / tat if tat > 0 else 0  # Equivalent Doorstep Frequency
            
            accessibility_results.append({
                'job_id': idx,
                'job_x': job_x,
                'job_y': job_y,
                'wt_minutes': wt,
                'awt_minutes': awt,
                'tat_minutes': tat,
                'edf': edf
            })
            
        # Convert to DataFrame
        self.accessibility_df = pd.DataFrame(accessibility_results)
        print(f"Calculated accessibility for {len(self.accessibility_df)} jobs")
        
        return self.accessibility_df
        
    def classify_ptal_bands(self):
        """Classify accessibility indices into PTAL bands (1-10)"""
        print("\nClassifying into PTAL bands...")
        
        if self.accessibility_df is None or len(self.accessibility_df) == 0:
            print("No accessibility data to classify")
            return None
            
        # Calculate accessibility index for each job (using EDF as proxy)
        ai_values = self.accessibility_df['edf'].values
        
        # Define PTAL band thresholds (based on EDF values)
        # Higher EDF = better accessibility = higher PTAL
        percentiles = np.percentile(ai_values, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        def assign_ptal(edf_value):
            if edf_value >= percentiles[9]:
                return 10
            elif edf_value >= percentiles[8]:
                return 9
            elif edf_value >= percentiles[7]:
                return 8
            elif edf_value >= percentiles[6]:
                return 7
            elif edf_value >= percentiles[5]:
                return 6
            elif edf_value >= percentiles[4]:
                return 5
            elif edf_value >= percentiles[3]:
                return 4
            elif edf_value >= percentiles[2]:
                return 3
            elif edf_value >= percentiles[1]:
                return 2
            else:
                return 1
                
        self.accessibility_df['ptal_band'] = self.accessibility_df['edf'].apply(assign_ptal)
        
        # Print PTAL distribution
        ptal_counts = self.accessibility_df['ptal_band'].value_counts().sort_index()
        print("\nPTAL Band Distribution:")
        for band, count in ptal_counts.items():
            percentage = (count / len(self.accessibility_df)) * 100
            print(f"PTAL {band}: {count} jobs ({percentage:.1f}%)")
            
        return self.accessibility_df
        
    def save_results(self):
        """Save accessibility calculation results"""
        print("\nSaving results...")
        
        if self.accessibility_df is None:
            print("No results to save")
            return
            
        # Save main results
        output_file = os.path.join(self.output_dir, "uam_accessibility_results.csv")
        self.accessibility_df.to_csv(output_file, index=False)
        print(f"Saved accessibility results to: {output_file}")
        
        # Save summary statistics
        summary_stats = {
            'total_jobs': len(self.accessibility_df),
            'mean_wt': self.accessibility_df['wt_minutes'].mean(),
            'mean_tat': self.accessibility_df['tat_minutes'].mean(),
            'mean_edf': self.accessibility_df['edf'].mean(),
            'mean_ptal': self.accessibility_df['ptal_band'].mean(),
            'awt_constant': self.awt_constant
        }
        
        summary_file = os.path.join(self.output_dir, "uam_accessibility_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("UAM Accessibility Calculation Summary\n")
            f.write("=====================================\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value:.2f}\n")
                
        print(f"Saved summary to: {summary_file}")
        
    def run_full_calculation(self):
        """Run the complete accessibility calculation pipeline"""
        print("Starting UAM PTAL-Style Accessibility Calculation")
        print("=" * 50)
        
        try:
            # Load data
            self.load_data()
            
            # Calculate accessibility
            accessibility_df = self.calculate_accessibility()
            
            if accessibility_df is not None:
                # Classify into PTAL bands
                classified_df = self.classify_ptal_bands()
                
                # Save results
                self.save_results()
                
                print("\n" + "=" * 50)
                print("UAM Accessibility Calculation Complete!")
                print(f"Results saved to: {self.output_dir}")
                
            else:
                print("Calculation failed - no accessibility data generated")
                
        except Exception as e:
            print(f"Error during calculation: {str(e)}")
            raise

def main():
    """Main function to run the accessibility calculation"""
    calculator = UAMAccessibilityCalculator()
    calculator.run_full_calculation()

if __name__ == "__main__":
    main()
