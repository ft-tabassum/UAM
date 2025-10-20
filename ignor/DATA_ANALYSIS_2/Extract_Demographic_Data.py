"""
Extract Demographic Data Used for Demographics Analysis PNG
This script extracts all the demographic data that was used to create
the demographic analysis visualization and saves it to CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_demographic_data():
    """Extract all demographic data used in the analysis"""
    
    print("="*80)
    print("EXTRACTING DEMOGRAPHIC DATA FOR ANALYSIS")
    print("="*80)
    
    # Load data
    file_path = 'D:/Files_D/Study/==Thesis==/new_data/aft_2ndversion.xlsx'
    
    try:
        data = pd.read_excel(file_path)
        print(f"[OK] Data loaded successfully")
        print(f"Total rows: {len(data)}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return
    
    # Create output directory
    output_dir = Path("../../../ignor/DATA_ANALYSIS/Main_Data_Analysis_2/Demographic_Data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract demographic columns
    demo_cols = [col for col in data.columns if any(x in col.lower() for x in ['age', 'gender', 'income', 'education', 'employment'])]
    print(f"\n1. DEMOGRAPHIC COLUMNS FOUND:")
    print(f"   Found {len(demo_cols)} demographic columns:")
    for i, col in enumerate(demo_cols, 1):
        print(f"   {i:2d}. {col}")
    
    # 2. Create comprehensive demographic table
    print(f"\n2. CREATING COMPREHENSIVE DEMOGRAPHIC TABLE:")
    
    # Select relevant columns for demographic analysis
    demographic_data = data[['sys_RespNum', 'CHOICE'] + demo_cols].copy()
    
    # Add choice labels
    choice_labels = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}
    demographic_data['CHOICE_LABEL'] = demographic_data['CHOICE'].map(choice_labels)
    
    # Save full demographic dataset
    demographic_file = output_dir / 'full_demographic_data.csv'
    demographic_data.to_csv(demographic_file, index=False)
    print(f"   [OK] Full demographic data saved to: {demographic_file}")
    print(f"   Shape: {demographic_data.shape}")
    
    # 3. Create cross-tabulation tables for each demographic variable
    print(f"\n3. CREATING CROSS-TABULATION TABLES:")
    
    for col in demo_cols:
        if col in data.columns:
            print(f"\n   Processing: {col}")
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(data[col], data['CHOICE'], margins=True)
            cross_tab.columns = ['Car', 'Public Transport', 'Flying Taxi', 'Total']
            cross_tab.index.name = col
            
            # Add percentages
            cross_tab_pct = pd.crosstab(data[col], data['CHOICE'], normalize='index') * 100
            cross_tab_pct.columns = ['Car_%', 'Public Transport_%', 'Flying Taxi_%']
            
            # Combine counts and percentages
            combined_tab = pd.concat([cross_tab, cross_tab_pct], axis=1)
            
            # Save to CSV (clean filename for Windows compatibility)
            clean_col_name = col.replace('<', 'lt_').replace('>', 'gt_').replace('+', 'plus_').replace('-', '_')
            filename = f'crosstab_{clean_col_name}.csv'
            filepath = output_dir / filename
            combined_tab.to_csv(filepath)
            print(f"   [OK] Saved: {filename}")
            print(f"   Shape: {combined_tab.shape}")
            
            # Print summary
            print(f"   Summary for {col}:")
            for choice in [1, 2, 3]:
                choice_name = choice_labels[choice]
                total_choice = cross_tab[choice_name].sum()
                print(f"     {choice_name}: {total_choice} total responses")
    
    # 4. Create summary statistics table
    print(f"\n4. CREATING SUMMARY STATISTICS TABLE:")
    
    summary_stats = []
    
    for col in demo_cols:
        if col in data.columns:
            # Basic statistics
            stats = {
                'Variable': col,
                'Data_Type': str(data[col].dtype),
                'Unique_Values': data[col].nunique(),
                'Min_Value': data[col].min(),
                'Max_Value': data[col].max(),
                'Mean_Value': data[col].mean() if data[col].dtype in ['int64', 'float64'] else 'N/A',
                'Std_Value': data[col].std() if data[col].dtype in ['int64', 'float64'] else 'N/A',
                'Missing_Values': data[col].isnull().sum(),
                'Missing_Percent': (data[col].isnull().sum() / len(data)) * 100
            }
            
            # Choice distribution
            for choice in [1, 2, 3]:
                choice_name = choice_labels[choice]
                choice_count = len(data[data['CHOICE'] == choice])
                var_choice_count = len(data[(data['CHOICE'] == choice) & (data[col] == 1)])
                stats[f'{choice_name}_Count'] = var_choice_count
                stats[f'{choice_name}_Percent'] = (var_choice_count / choice_count) * 100 if choice_count > 0 else 0
            
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = output_dir / 'demographic_summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"   [OK] Summary statistics saved to: {summary_file}")
    
    # 5. Create choice distribution by demographics
    print(f"\n5. CREATING CHOICE DISTRIBUTION BY DEMOGRAPHICS:")
    
    choice_demo_analysis = []
    
    for col in demo_cols:
        if col in data.columns:
            # Get unique values for this demographic variable
            unique_vals = sorted(data[col].unique())
            
            for val in unique_vals:
                if val == 1:  # Only analyze when the demographic condition is true (1)
                    subset = data[data[col] == val]
                    total_in_group = len(subset)
                    
                    if total_in_group > 0:
                        choice_dist = subset['CHOICE'].value_counts().sort_index()
                        
                        analysis = {
                            'Demographic_Variable': col,
                            'Demographic_Value': val,
                            'Total_in_Group': total_in_group,
                            'Car_Count': choice_dist.get(1, 0),
                            'Public_Transport_Count': choice_dist.get(2, 0),
                            'Flying_Taxi_Count': choice_dist.get(3, 0),
                            'Car_Percent': (choice_dist.get(1, 0) / total_in_group) * 100,
                            'Public_Transport_Percent': (choice_dist.get(2, 0) / total_in_group) * 100,
                            'Flying_Taxi_Percent': (choice_dist.get(3, 0) / total_in_group) * 100
                        }
                        
                        choice_demo_analysis.append(analysis)
    
    choice_demo_df = pd.DataFrame(choice_demo_analysis)
    choice_demo_file = output_dir / 'choice_distribution_by_demographics.csv'
    choice_demo_df.to_csv(choice_demo_file, index=False)
    print(f"   [OK] Choice distribution by demographics saved to: {choice_demo_file}")
    
    # 6. Create age group analysis (special case)
    print(f"\n6. CREATING AGE GROUP ANALYSIS:")
    
    age_cols = [col for col in demo_cols if 'age' in col]
    age_analysis = []
    
    for col in age_cols:
        if col in data.columns:
            # Count people in this age group
            in_age_group = data[data[col] == 1]
            not_in_age_group = data[data[col] == 0]
            
            # Choice distribution for people IN this age group
            if len(in_age_group) > 0:
                choice_dist_in = in_age_group['CHOICE'].value_counts().sort_index()
                analysis_in = {
                    'Age_Group': col,
                    'Group_Status': 'In_Age_Group',
                    'Total_Count': len(in_age_group),
                    'Car_Count': choice_dist_in.get(1, 0),
                    'Public_Transport_Count': choice_dist_in.get(2, 0),
                    'Flying_Taxi_Count': choice_dist_in.get(3, 0),
                    'Car_Percent': (choice_dist_in.get(1, 0) / len(in_age_group)) * 100,
                    'Public_Transport_Percent': (choice_dist_in.get(2, 0) / len(in_age_group)) * 100,
                    'Flying_Taxi_Percent': (choice_dist_in.get(3, 0) / len(in_age_group)) * 100
                }
                age_analysis.append(analysis_in)
            
            # Choice distribution for people NOT in this age group
            if len(not_in_age_group) > 0:
                choice_dist_not = not_in_age_group['CHOICE'].value_counts().sort_index()
                analysis_not = {
                    'Age_Group': col,
                    'Group_Status': 'Not_In_Age_Group',
                    'Total_Count': len(not_in_age_group),
                    'Car_Count': choice_dist_not.get(1, 0),
                    'Public_Transport_Count': choice_dist_not.get(2, 0),
                    'Flying_Taxi_Count': choice_dist_not.get(3, 0),
                    'Car_Percent': (choice_dist_not.get(1, 0) / len(not_in_age_group)) * 100,
                    'Public_Transport_Percent': (choice_dist_not.get(2, 0) / len(not_in_age_group)) * 100,
                    'Flying_Taxi_Percent': (choice_dist_not.get(3, 0) / len(not_in_age_group)) * 100
                }
                age_analysis.append(analysis_not)
    
    age_analysis_df = pd.DataFrame(age_analysis)
    age_analysis_file = output_dir / 'age_group_analysis.csv'
    age_analysis_df.to_csv(age_analysis_file, index=False)
    print(f"   [OK] Age group analysis saved to: {age_analysis_file}")
    
    # 7. Create overall choice summary
    print(f"\n7. CREATING OVERALL CHOICE SUMMARY:")
    
    overall_choice = data['CHOICE'].value_counts().sort_index()
    overall_summary = pd.DataFrame({
        'Transport_Mode': ['Car', 'Public Transport', 'Flying Taxi'],
        'Choice_Code': [1, 2, 3],
        'Count': [overall_choice.get(1, 0), overall_choice.get(2, 0), overall_choice.get(3, 0)],
        'Percentage': [(overall_choice.get(1, 0) / len(data)) * 100, 
                      (overall_choice.get(2, 0) / len(data)) * 100, 
                      (overall_choice.get(3, 0) / len(data)) * 100]
    })
    
    overall_file = output_dir / 'overall_choice_summary.csv'
    overall_summary.to_csv(overall_file, index=False)
    print(f"   [OK] Overall choice summary saved to: {overall_file}")
    
    # 8. Print summary of all files created
    print(f"\n8. SUMMARY OF FILES CREATED:")
    print(f"   Output directory: {output_dir}")
    print(f"   Files created:")
    
    files_created = [
        'full_demographic_data.csv',
        'demographic_summary_statistics.csv', 
        'choice_distribution_by_demographics.csv',
        'age_group_analysis.csv',
        'overall_choice_summary.csv'
    ]
    
    for filename in files_created:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"   [OK] {filename}")
        else:
            print(f"   [MISSING] {filename} (not found)")
    
    # Add cross-tabulation files
    for col in demo_cols:
        clean_col_name = col.replace('<', 'lt_').replace('>', 'gt_').replace('+', 'plus_').replace('-', '_')
        filename = f'crosstab_{clean_col_name}.csv'
        filepath = output_dir / filename
        if filepath.exists():
            print(f"   [OK] {filename}")
    
    print(f"\n[SUCCESS] All demographic data extracted successfully!")
    print(f"[SUCCESS] Files saved to: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    extract_demographic_data()
