"""
Percentage Analysis Table Generator
Creates detailed percentage tables for Environmental Concern, Technology Concern, and Likelihood variables
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_percentage_tables():
    """Create percentage distribution tables for key variables"""
    
    # Load the data
    file_path = 'D:/Files_D/Study/==Thesis==/new_data/aft_2ndversion.xlsx'
    data = pd.read_excel(file_path)
    
    print("="*80)
    print("PERCENTAGE DISTRIBUTION TABLES")
    print("="*80)
    
    # Create output directory
    output_dir = Path("../DATA_ANALYSIS/Main_Data_Analysis_2/simpleMain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ENVIRONMENTAL CONCERN VARIABLES
    print("\n1. ENVIRONMENTAL CONCERN VARIABLES - PERCENTAGE DISTRIBUTION")
    print("="*60)
    
    env_concern_cols = [col for col in data.columns if 'environmentconcern' in col]
    env_data = []
    
    for col in env_concern_cols:
        value_counts = data[col].value_counts().sort_index()
        percentages = (value_counts / len(data)) * 100
        
        # Get variable meaning
        meanings = {
            'environmentconcern_r1': 'Concerned about Global Warming',
            'environmentconcern_r2': 'Do NOT change behavior for environment',
            'environmentconcern_r3': 'Acceptable to cause pollution',
            'environmentconcern_r4': 'Willing to pay more for eco-friendly products'
        }
        
        meaning = meanings.get(col, col)
        print(f"\n{col} - {meaning}")
        print("-" * 50)
        
        for value, count in value_counts.items():
            percentage = percentages[value]
            print(f"  {value}: {count:4d} ({percentage:5.1f}%)")
            
            env_data.append({
                'Variable': col,
                'Meaning': meaning,
                'Value': value,
                'Count': count,
                'Percentage': percentage
            })
    
    # 2. TECHNOLOGY CONCERN VARIABLES
    print("\n\n2. TECHNOLOGY CONCERN VARIABLES - PERCENTAGE DISTRIBUTION")
    print("="*60)
    
    tech_concern_cols = [col for col in data.columns if 'technologyconcern' in col]
    tech_data = []
    
    for col in tech_concern_cols:
        value_counts = data[col].value_counts().sort_index()
        percentages = (value_counts / len(data)) * 100
        
        # Get variable meaning
        meanings = {
            'technologyconcern_r1': 'Excited by new technology',
            'technologyconcern_r2': 'Use expensive new technology',
            'technologyconcern_r3': 'Little interest in technology',
            'technologyconcern_r4': 'Technology causes problems'
        }
        
        meaning = meanings.get(col, col)
        print(f"\n{col} - {meaning}")
        print("-" * 50)
        
        for value, count in value_counts.items():
            percentage = percentages[value]
            print(f"  {value}: {count:4d} ({percentage:5.1f}%)")
            
            tech_data.append({
                'Variable': col,
                'Meaning': meaning,
                'Value': value,
                'Count': count,
                'Percentage': percentage
            })
    
    # 3. LIKELIHOOD VARIABLES (Travel Purpose)
    print("\n\n3. LIKELIHOOD VARIABLES (Travel Purpose) - PERCENTAGE DISTRIBUTION")
    print("="*60)
    
    likelihood_cols = [col for col in data.columns if 'Likelihood' in col]
    likelihood_data = []
    
    for col in likelihood_cols:
        value_counts = data[col].value_counts().sort_index()
        percentages = (value_counts / len(data)) * 100
        
        # Get variable meaning
        meanings = {
            'Likelihood_r1': 'Travel Purpose Scenario 1',
            'Likelihood_r2': 'Travel Purpose Scenario 2', 
            'Likelihood_r3': 'Travel Purpose Scenario 3',
            'Likelihood_r4': 'Travel Purpose Scenario 4',
            'Likelihood_r5': 'Travel Purpose Scenario 5',
            'Likelihood_r6': 'Travel Purpose Scenario 6'
        }
        
        meaning = meanings.get(col, col)
        print(f"\n{col} - {meaning}")
        print("-" * 50)
        
        for value, count in value_counts.items():
            percentage = percentages[value]
            purpose = {1: "Work", 2: "Business", 3: "Shopping", 4: "Recreational", 5: "Education", 6: "Social"}.get(value, f"Unknown ({value})")
            print(f"  {value} ({purpose}): {count:4d} ({percentage:5.1f}%)")
            
            likelihood_data.append({
                'Variable': col,
                'Meaning': meaning,
                'Value': value,
                'Purpose': purpose,
                'Count': count,
                'Percentage': percentage
            })
    
    # Create summary tables
    print("\n\n4. SUMMARY TABLES")
    print("="*60)
    
    # Environmental Concern Summary
    env_df = pd.DataFrame(env_data)
    env_summary = env_df.pivot_table(
        index=['Variable', 'Meaning'], 
        columns='Value', 
        values='Percentage', 
        fill_value=0
    ).round(1)
    
    print("\nEnvironmental Concern - Percentage Summary:")
    print(env_summary)
    
    # Technology Concern Summary
    tech_df = pd.DataFrame(tech_data)
    tech_summary = tech_df.pivot_table(
        index=['Variable', 'Meaning'], 
        columns='Value', 
        values='Percentage', 
        fill_value=0
    ).round(1)
    
    print("\nTechnology Concern - Percentage Summary:")
    print(tech_summary)
    
    # Likelihood Summary
    likelihood_df = pd.DataFrame(likelihood_data)
    likelihood_summary = likelihood_df.pivot_table(
        index=['Variable', 'Meaning'], 
        columns='Purpose', 
        values='Percentage', 
        fill_value=0
    ).round(1)
    
    print("\nLikelihood (Travel Purpose) - Percentage Summary:")
    print(likelihood_summary)
    
    # Save to files
    print(f"\n5. SAVING RESULTS TO: {output_dir}")
    
    # Save detailed data
    env_df.to_csv(output_dir / 'environmental_concern_percentages.csv', index=False)
    tech_df.to_csv(output_dir / 'technology_concern_percentages.csv', index=False)
    likelihood_df.to_csv(output_dir / 'likelihood_percentages.csv', index=False)
    
    # Save summary tables
    env_summary.to_csv(output_dir / 'environmental_concern_summary.csv')
    tech_summary.to_csv(output_dir / 'technology_concern_summary.csv')
    likelihood_summary.to_csv(output_dir / 'likelihood_summary.csv')
    
    # Create a comprehensive table
    create_comprehensive_table(env_df, tech_df, likelihood_df, output_dir)
    
    print("✓ All percentage tables saved successfully!")
    
    return env_df, tech_df, likelihood_df

def create_comprehensive_table(env_df, tech_df, likelihood_df, output_dir):
    """Create a comprehensive table combining all variables"""
    
    print("\n6. CREATING COMPREHENSIVE TABLE")
    print("-" * 40)
    
    # Combine all data
    all_data = []
    
    # Add environmental concern data
    for _, row in env_df.iterrows():
        all_data.append({
            'Category': 'Environmental Concern',
            'Variable': row['Variable'],
            'Meaning': row['Meaning'],
            'Value': row['Value'],
            'Count': row['Count'],
            'Percentage': row['Percentage']
        })
    
    # Add technology concern data
    for _, row in tech_df.iterrows():
        all_data.append({
            'Category': 'Technology Concern',
            'Variable': row['Variable'],
            'Meaning': row['Meaning'],
            'Value': row['Value'],
            'Count': row['Count'],
            'Percentage': row['Percentage']
        })
    
    # Add likelihood data
    for _, row in likelihood_df.iterrows():
        all_data.append({
            'Category': 'Likelihood (Travel Purpose)',
            'Variable': row['Variable'],
            'Meaning': row['Meaning'],
            'Value': f"{row['Value']} ({row['Purpose']})",
            'Count': row['Count'],
            'Percentage': row['Percentage']
        })
    
    # Create comprehensive DataFrame
    comprehensive_df = pd.DataFrame(all_data)
    
    # Save comprehensive table
    comprehensive_df.to_csv(output_dir / 'comprehensive_percentage_table.csv', index=False)
    
    # Create pivot table for easy viewing
    pivot_table = comprehensive_df.pivot_table(
        index=['Category', 'Variable', 'Meaning'],
        columns='Value',
        values='Percentage',
        fill_value=0
    ).round(1)
    
    pivot_table.to_csv(output_dir / 'comprehensive_percentage_pivot.csv')
    
    print("✓ Comprehensive table created and saved!")
    
    return comprehensive_df

if __name__ == "__main__":
    try:
        env_df, tech_df, likelihood_df = create_percentage_tables()
        print(f"\n✓ Analysis completed successfully!")
        print(f"✓ Check the output directory for all CSV files")
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
