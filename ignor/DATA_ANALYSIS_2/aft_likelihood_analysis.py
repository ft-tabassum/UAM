"""
AFT Likelihood Analysis - Trip Purposes
Analyzes the likelihood of choosing AFT (Autonomous Flying Taxi) for different trip purposes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_aft_likelihood_by_trip_purpose():
    """Analyze AFT choice likelihood by different trip purposes"""
    
    # Load the data
    file_path = 'D:/Files_D/Study/==Thesis==/new_data/aft_2ndversion.xlsx'
    data = pd.read_excel(file_path)
    
    print("="*80)
    print("LIKELIHOOD OF CHOOSING AFT REGARDING DIFFERENT TRIP PURPOSES")
    print("="*80)
    
    # Create output directory
    output_dir = Path("../DATA_ANALYSIS/Main_Data_Analysis_2/simpleMain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get likelihood variables (trip purposes)
    likelihood_cols = [col for col in data.columns if 'Likelihood' in col]
    
    # Trip purpose meanings
    trip_purposes = {
        1: "Work",
        2: "Business", 
        3: "Shopping",
        4: "Recreational",
        5: "Education",
        6: "Social"
    }
    
    print(f"\n1. OVERVIEW OF LIKELIHOOD VARIABLES")
    print("-" * 50)
    print(f"Found {len(likelihood_cols)} likelihood scenarios:")
    for i, col in enumerate(likelihood_cols, 1):
        print(f"  {i}. {col}")
    
    # Analyze each likelihood scenario
    print(f"\n2. DETAILED ANALYSIS BY TRIP PURPOSE")
    print("="*60)
    
    likelihood_analysis = []
    
    for col in likelihood_cols:
        print(f"\n{col}:")
        print("-" * 30)
        
        # Get value counts and percentages
        value_counts = data[col].value_counts().sort_index()
        percentages = (value_counts / len(data)) * 100
        
        for value, count in value_counts.items():
            percentage = percentages[value]
            purpose = trip_purposes.get(value, f"Unknown ({value})")
            print(f"  {value} ({purpose}): {count:4d} ({percentage:5.1f}%)")
            
            likelihood_analysis.append({
                'Scenario': col,
                'Value': value,
                'Trip_Purpose': purpose,
                'Count': count,
                'Percentage': percentage
            })
    
    # Create summary table
    print(f"\n3. SUMMARY TABLE - TRIP PURPOSE DISTRIBUTION")
    print("="*60)
    
    likelihood_df = pd.DataFrame(likelihood_analysis)
    
    # Create pivot table for easy viewing
    pivot_table = likelihood_df.pivot_table(
        index='Trip_Purpose',
        columns='Scenario', 
        values='Percentage',
        fill_value=0
    ).round(1)
    
    print(pivot_table)
    
    # Calculate average percentages for each trip purpose across all scenarios
    print(f"\n4. AVERAGE LIKELIHOOD BY TRIP PURPOSE")
    print("-" * 50)
    
    avg_by_purpose = likelihood_df.groupby('Trip_Purpose')['Percentage'].mean().sort_values(ascending=False)
    
    for purpose, avg_percentage in avg_by_purpose.items():
        print(f"  {purpose:15s}: {avg_percentage:5.1f}%")
    
    # Analyze AFT choice correlation with trip purposes
    print(f"\n5. AFT CHOICE vs TRIP PURPOSE ANALYSIS")
    print("="*60)
    
    if 'CHOICE' in data.columns:
        # Filter for AFT choice (choice = 3)
        aft_data = data[data['CHOICE'] == 3]
        print(f"Total AFT choices: {len(aft_data)} ({len(aft_data)/len(data)*100:.1f}% of total)")
        
        print(f"\nAFT Choice Distribution by Trip Purpose:")
        print("-" * 40)
        
        aft_trip_analysis = []
        
        for col in likelihood_cols:
            # Get trip purpose distribution for AFT choosers
            aft_value_counts = aft_data[col].value_counts().sort_index()
            aft_percentages = (aft_value_counts / len(aft_data)) * 100
            
            print(f"\n{col} (AFT Choosers):")
            for value, count in aft_value_counts.items():
                percentage = aft_percentages[value]
                purpose = trip_purposes.get(value, f"Unknown ({value})")
                print(f"  {value} ({purpose}): {count:3d} ({percentage:5.1f}%)")
                
                aft_trip_analysis.append({
                    'Scenario': col,
                    'Value': value,
                    'Trip_Purpose': purpose,
                    'Count': count,
                    'Percentage': percentage,
                    'Type': 'AFT_Choosers'
                })
        
        # Compare with overall population
        print(f"\nComparison: AFT Choosers vs Overall Population")
        print("-" * 50)
        
        comparison_data = []
        
        for col in likelihood_cols:
            # Overall population percentages
            overall_counts = data[col].value_counts().sort_index()
            overall_percentages = (overall_counts / len(data)) * 100
            
            # AFT choosers percentages
            aft_counts = aft_data[col].value_counts().sort_index()
            aft_percentages = (aft_counts / len(aft_data)) * 100
            
            print(f"\n{col}:")
            for value in sorted(overall_counts.index):
                purpose = trip_purposes.get(value, f"Unknown ({value})")
                overall_pct = overall_percentages[value]
                aft_pct = aft_percentages.get(value, 0)
                difference = aft_pct - overall_pct
                
                print(f"  {value} ({purpose}):")
                print(f"    Overall: {overall_pct:5.1f}% | AFT: {aft_pct:5.1f}% | Diff: {difference:+5.1f}%")
                
                comparison_data.append({
                    'Scenario': col,
                    'Value': value,
                    'Trip_Purpose': purpose,
                    'Overall_Percentage': overall_pct,
                    'AFT_Percentage': aft_pct,
                    'Difference': difference
                })
    
    # Create visualizations
    print(f"\n6. CREATING VISUALIZATIONS")
    print("-" * 30)
    create_visualizations(likelihood_df, output_dir)
    
    # Save results
    print(f"\n7. SAVING RESULTS")
    print("-" * 20)
    
    # Save detailed analysis
    likelihood_df.to_csv(output_dir / 'aft_likelihood_detailed.csv', index=False)
    pivot_table.to_csv(output_dir / 'aft_likelihood_summary.csv')
    
    if 'CHOICE' in data.columns:
        aft_trip_df = pd.DataFrame(aft_trip_analysis)
        comparison_df = pd.DataFrame(comparison_data)
        
        aft_trip_df.to_csv(output_dir / 'aft_choosers_trip_purpose.csv', index=False)
        comparison_df.to_csv(output_dir / 'aft_vs_overall_comparison.csv', index=False)
    
    # Create final summary table
    create_final_summary_table(likelihood_df, output_dir)
    
    print(f"✓ All results saved to: {output_dir}")
    
    return likelihood_df

def create_visualizations(likelihood_df, output_dir):
    """Create visualizations for AFT likelihood analysis"""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Trip Purpose Distribution
    plt.figure(figsize=(15, 10))
    
    # Get unique trip purposes and their average percentages
    avg_by_purpose = likelihood_df.groupby('Trip_Purpose')['Percentage'].mean().sort_values(ascending=True)
    
    plt.subplot(2, 2, 1)
    bars = avg_by_purpose.plot(kind='barh', color='skyblue', edgecolor='navy')
    plt.title('Average Likelihood by Trip Purpose', fontsize=14, fontweight='bold')
    plt.xlabel('Average Percentage (%)')
    plt.ylabel('Trip Purpose')
    
    # Add value labels on bars
    for i, v in enumerate(avg_by_purpose.values):
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # 2. Scenario Comparison
    plt.subplot(2, 2, 2)
    pivot_data = likelihood_df.pivot_table(
        index='Trip_Purpose',
        columns='Scenario',
        values='Percentage',
        fill_value=0
    )
    
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Percentage (%)'})
    plt.title('Trip Purpose Distribution by Scenario', fontsize=14, fontweight='bold')
    plt.xlabel('Likelihood Scenario')
    plt.ylabel('Trip Purpose')
    
    # 3. Trip Purpose Distribution (Stacked Bar)
    plt.subplot(2, 2, 3)
    pivot_data.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Trip Purpose Distribution by Scenario', fontsize=14, fontweight='bold')
    plt.xlabel('Trip Purpose')
    plt.ylabel('Percentage (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # 4. Summary Statistics
    plt.subplot(2, 2, 4)
    summary_stats = likelihood_df.groupby('Trip_Purpose')['Percentage'].agg(['mean', 'std', 'min', 'max'])
    
    x_pos = range(len(summary_stats))
    plt.errorbar(x_pos, summary_stats['mean'], yerr=summary_stats['std'], 
                fmt='o', capsize=5, capthick=2, markersize=8)
    plt.title('Trip Purpose Statistics (Mean ± Std)', fontsize=14, fontweight='bold')
    plt.xlabel('Trip Purpose')
    plt.ylabel('Percentage (%)')
    plt.xticks(x_pos, summary_stats.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aft_likelihood_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualizations saved!")

def create_final_summary_table(likelihood_df, output_dir):
    """Create a final summary table for easy reference"""
    
    print("Creating final summary table...")
    
    # Create comprehensive summary
    summary_data = []
    
    # Get average by trip purpose
    avg_by_purpose = likelihood_df.groupby('Trip_Purpose')['Percentage'].agg(['mean', 'std', 'min', 'max']).round(1)
    
    for purpose in avg_by_purpose.index:
        summary_data.append({
            'Trip_Purpose': purpose,
            'Average_Percentage': avg_by_purpose.loc[purpose, 'mean'],
            'Standard_Deviation': avg_by_purpose.loc[purpose, 'std'],
            'Minimum_Percentage': avg_by_purpose.loc[purpose, 'min'],
            'Maximum_Percentage': avg_by_purpose.loc[purpose, 'max'],
            'Rank': list(avg_by_purpose['mean'].sort_values(ascending=False).index).index(purpose) + 1
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Average_Percentage', ascending=False)
    
    # Save summary
    summary_df.to_csv(output_dir / 'aft_likelihood_final_summary.csv', index=False)
    
    print("Final Summary Table:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    try:
        likelihood_df = analyze_aft_likelihood_by_trip_purpose()
        print(f"\n✓ AFT Likelihood Analysis completed successfully!")
        print(f"✓ Check the output directory for all analysis files")
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
