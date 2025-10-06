"""
Simple Main Data Analysis Script for AFT Survey Data
This script analyzes the raw survey data focusing on key variables
and produces output similar to the processed CSV format.

Key Variables Analyzed:
- choice: 1=car, 2=public transport, 3=flying taxi
- Likelihood: Travel purpose (1=Work, 2=Business, 3=Shopping, 4=Recreational, 5=Education, 6=Social)
- AtoLattitude: Attitude towards autonomous technology
- technologyconcern: Technology concern
- environmentconcern: Environmental concern
- co: Travel cost
- INC: Inconvenience (walking/waiting time)
- MULTI: Multitasking possibility
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_main_data(file_path):
    """
    Analyze the main survey data file
    
    Args:
        file_path (str): Path to the Excel file containing survey data
    
    Returns:
        pd.DataFrame: Analyzed data with key insights
    """
    
    print("="*80)
    print("SIMPLE MAIN DATA ANALYSIS FOR AFT SURVEY")
    print("="*80)
    
    # Load data
    print(f"\n1. LOADING DATA FROM: {file_path}")
    try:
        data = pd.read_excel(file_path)
        print(f"✓ Data loaded successfully")
        print(f"  - Shape: {data.shape}")
        print(f"  - Columns: {len(data.columns)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Show column names
    print(f"\n2. COLUMN NAMES:")
    for i, col in enumerate(data.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Analyze CHOICE variable (target)
    print(f"\n3. TARGET VARIABLE ANALYSIS (CHOICE):")
    if 'CHOICE' in data.columns:
        print(f"   - Data type: {data['CHOICE'].dtype}")
        print(f"   - Unique values: {sorted(data['CHOICE'].unique())}")
        print(f"   - Distribution:")
        choice_counts = data['CHOICE'].value_counts().sort_index()
        for choice, count in choice_counts.items():
            percentage = (count / len(data)) * 100
            choice_name = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}.get(choice, f"Unknown ({choice})")
            print(f"     {choice} ({choice_name}): {count} ({percentage:.1f}%)")
    else:
        print("   ✗ CHOICE column not found")
    
    # Analyze key variables as described by user
    print(f"\n4. KEY VARIABLES ANALYSIS:")
    
    # Find and analyze Likelihood variables (travel purpose)
    likelihood_cols = [col for col in data.columns if 'Likelihood' in col or 'likelihood' in col]
    if likelihood_cols:
        print(f"\n   a) LIKELIHOOD VARIABLES (Travel Purpose):")
        print(f"      Found: {likelihood_cols}")
        for col in likelihood_cols:
            if col in data.columns:
                unique_vals = sorted(data[col].unique())
                print(f"      - {col}:")
                print(f"        Values: {unique_vals}")
                print(f"        Count: {data[col].value_counts().sort_index().to_dict()}")
                # Map values to meanings
                if 1 in unique_vals:
                    print(f"        Meaning: 1=Work, 2=Business, 3=Shopping, 4=Recreational, 5=Education, 6=Social")
    
    # Find and analyze Attitude towards autonomous technology
    attitude_cols = [col for col in data.columns if 'AtoLattitude' in col or 'attitude' in col.lower()]
    if attitude_cols:
        print(f"\n   b) ATTITUDE TOWARDS AUTONOMOUS TECHNOLOGY:")
        print(f"      Found: {attitude_cols}")
        for col in attitude_cols:
            if col in data.columns:
                unique_vals = sorted(data[col].unique())
                print(f"      - {col}:")
                print(f"        Values: {unique_vals}")
                print(f"        Count: {data[col].value_counts().sort_index().to_dict()}")
                # Map values to meanings
                if 1 in unique_vals:
                    print(f"        Meaning: 1=Fun to use, 2=Fear of self-driving taxi, 3=Important role, 4=Fear of flying taxi")
    
    # Find and analyze Technology concern
    tech_concern_cols = [col for col in data.columns if 'technologyconcern' in col or 'technology' in col.lower()]
    if tech_concern_cols:
        print(f"\n   c) TECHNOLOGY CONCERN:")
        print(f"      Found: {tech_concern_cols}")
        for col in tech_concern_cols:
            if col in data.columns:
                unique_vals = sorted(data[col].unique())
                print(f"      - {col}:")
                print(f"        Values: {unique_vals}")
                print(f"        Count: {data[col].value_counts().sort_index().to_dict()}")
                # Map values to meanings
                if 1 in unique_vals:
                    print(f"        Meaning: 1=Excited by new tech, 2=Use expensive new tech, 3=Little interest, 4=Tech causes problems")
    
    # Find and analyze Environmental concern
    env_concern_cols = [col for col in data.columns if 'environmentconcern' in col or 'environment' in col.lower()]
    if env_concern_cols:
        print(f"\n   d) ENVIRONMENTAL CONCERN:")
        print(f"      Found: {env_concern_cols}")
        for col in env_concern_cols:
            if col in data.columns:
                unique_vals = sorted(data[col].unique())
                print(f"      - {col}:")
                print(f"        Values: {unique_vals}")
                print(f"        Count: {data[col].value_counts().sort_index().to_dict()}")
                # Map values to meanings
                if 1 in unique_vals:
                    print(f"        Meaning: 1=Concerned about global warming, 2=Don't change behavior, 3=Acceptable pollution, 4=Willing to pay more for eco-friendly")
    
    # Find and analyze Cost variables
    cost_cols = [col for col in data.columns if 'co' in col.lower() or 'cost' in col.lower()]
    if cost_cols:
        print(f"\n   e) COST VARIABLES:")
        print(f"      Found: {cost_cols}")
        for col in cost_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Min: {data[col].min():.2f}")
                print(f"        Max: {data[col].max():.2f}")
                print(f"        Mean: {data[col].mean():.2f}")
                print(f"        Std: {data[col].std():.2f}")
                print(f"        Sample values: {data[col].head(5).tolist()}")
    
    # Find and analyze Inconvenience variables
    inc_cols = [col for col in data.columns if 'inc' in col.lower() or 'inconvenience' in col.lower()]
    if inc_cols:
        print(f"\n   f) INCONVENIENCE VARIABLES (Walking/Waiting Time):")
        print(f"      Found: {inc_cols}")
        for col in inc_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Min: {data[col].min():.2f}")
                print(f"        Max: {data[col].max():.2f}")
                print(f"        Mean: {data[col].mean():.2f}")
                print(f"        Std: {data[col].std():.2f}")
                print(f"        Sample values: {data[col].head(5).tolist()}")
    
    # Find and analyze Multitasking variables
    multi_cols = [col for col in data.columns if 'multi' in col.lower() or 'multitasking' in col.lower()]
    if multi_cols:
        print(f"\n   g) MULTITASKING VARIABLES:")
        print(f"      Found: {multi_cols}")
        for col in multi_cols:
            if col in data.columns:
                unique_vals = sorted(data[col].unique())
                print(f"      - {col}:")
                print(f"        Values: {unique_vals}")
                print(f"        Count: {data[col].value_counts().sort_index().to_dict()}")
    
    # Analyze relationship between key variables and choice
    if 'CHOICE' in data.columns:
        print(f"\n5. RELATIONSHIP WITH TRANSPORT CHOICE:")
        
        # Cost vs Choice
        if cost_cols:
            print(f"\n   Cost vs Choice Analysis:")
            for col in cost_cols:
                if col in data.columns:
                    print(f"   - {col} by Choice:")
                    cost_by_choice = data.groupby('CHOICE')[col].agg(['mean', 'std', 'min', 'max'])
                    for choice in sorted(data['CHOICE'].unique()):
                        choice_name = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}.get(choice, f"Unknown ({choice})")
                        mean_val = cost_by_choice.loc[choice, 'mean']
                        std_val = cost_by_choice.loc[choice, 'std']
                        print(f"     {choice} ({choice_name}): Mean={mean_val:.2f}, Std={std_val:.2f}")
        
        # Attitude vs Choice
        if attitude_cols:
            print(f"\n   Attitude vs Choice Analysis:")
            for col in attitude_cols:
                if col in data.columns:
                    print(f"   - {col} by Choice:")
                    attitude_by_choice = data.groupby('CHOICE')[col].agg(['mean', 'std'])
                    for choice in sorted(data['CHOICE'].unique()):
                        choice_name = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}.get(choice, f"Unknown ({choice})")
                        mean_val = attitude_by_choice.loc[choice, 'mean']
                        std_val = attitude_by_choice.loc[choice, 'std']
                        print(f"     {choice} ({choice_name}): Mean={mean_val:.2f}, Std={std_val:.2f}")
    
    # Data quality check
    print(f"\n6. DATA QUALITY CHECK:")
    missing_data = data.isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    
    print(f"   - Missing values:")
    for col in data.columns:
        if missing_data[col] > 0:
            print(f"     {col}: {missing_data[col]} ({missing_percent[col]:.1f}%)")
    
    if missing_data.sum() == 0:
        print("     ✓ No missing values found")
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"   - Duplicate rows: {duplicates}")
    
    # Summary
    print(f"\n7. SUMMARY:")
    print(f"   - Total observations: {len(data)}")
    print(f"   - Total features: {len(data.columns)}")
    print(f"   - Numeric features: {len(data.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categorical features: {len(data.select_dtypes(include=['object', 'category']).columns)}")
    
    # Save results
    print(f"\n8. SAVING RESULTS:")
    save_results(data, file_path)
    
    return data

def save_results(data, file_path):
    """Save analysis results"""
    
    # Create output directory
    output_dir = Path("Result/DataPreprocessing_aft/Main_Data_Analysis/simpleMain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data summary
    summary_file = output_dir / 'main_data_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MAIN DATA ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data file: {file_path}\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Total observations: {len(data)}\n")
        f.write(f"Total features: {len(data.columns)}\n\n")
        
        f.write("COLUMN INFORMATION:\n")
        f.write("-"*30 + "\n")
        for i, col in enumerate(data.columns, 1):
            f.write(f"{i:2d}. {col} ({data[col].dtype})\n")
        
        f.write("\nTARGET VARIABLE (CHOICE):\n")
        f.write("-"*30 + "\n")
        if 'CHOICE' in data.columns:
            choice_counts = data['CHOICE'].value_counts().sort_index()
            for choice, count in choice_counts.items():
                percentage = (count / len(data)) * 100
                choice_name = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}.get(choice, f"Unknown ({choice})")
                f.write(f"{choice} ({choice_name}): {count} ({percentage:.1f}%)\n")
        
        f.write("\nKEY VARIABLES SUMMARY:\n")
        f.write("-"*30 + "\n")
        
        # Likelihood
        likelihood_cols = [col for col in data.columns if 'Likelihood' in col]
        if likelihood_cols:
            f.write(f"Likelihood variables: {likelihood_cols}\n")
        
        # Attitude
        attitude_cols = [col for col in data.columns if 'AtoLattitude' in col]
        if attitude_cols:
            f.write(f"Attitude variables: {attitude_cols}\n")
        
        # Technology concern
        tech_concern_cols = [col for col in data.columns if 'technologyconcern' in col]
        if tech_concern_cols:
            f.write(f"Technology concern variables: {tech_concern_cols}\n")
        
        # Environmental concern
        env_concern_cols = [col for col in data.columns if 'environmentconcern' in col]
        if env_concern_cols:
            f.write(f"Environmental concern variables: {env_concern_cols}\n")
        
        # Cost
        cost_cols = [col for col in data.columns if 'co' in col.lower()]
        if cost_cols:
            f.write(f"Cost variables: {cost_cols}\n")
        
        # Inconvenience
        inc_cols = [col for col in data.columns if 'inc' in col.lower()]
        if inc_cols:
            f.write(f"Inconvenience variables: {inc_cols}\n")
        
        # Multitasking
        multi_cols = [col for col in data.columns if 'multi' in col.lower()]
        if multi_cols:
            f.write(f"Multitasking variables: {multi_cols}\n")
    
    # Save descriptive statistics
    stats_file = output_dir / 'main_data_statistics.csv'
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_stats = data[numeric_cols].describe()
        desc_stats.to_csv(stats_file)
    
    print(f"   ✓ Results saved to: {output_dir}")

def main():
    """Main function"""
    
    # File path - user will need to update this
    file_path = 'D:/Files_D/Study/==Thesis==/new_data/aft_2ndversion.xlsx'
    
    print("SIMPLE MAIN DATA ANALYSIS SCRIPT FOR AFT SURVEY")
    print("="*80)
    print(f"Target file: {file_path}")
    print("="*80)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"\n⚠️  WARNING: File not found at {file_path}")
        print("Please update the file_path variable in the script with the correct path.")
        print("\nTo use this script:")
        print("1. Update the file_path variable with your actual file location")
        print("2. Run the script: python Simple_Main_Data_Analysis.py")
        return
    
    # Run analysis
    try:
        data = analyze_main_data(file_path)
        if data is not None:
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: Result/DataPreprocessing_aft/Main_Data_Analysis/")
        else:
            print(f"\n✗ Analysis failed!")
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
