"""
Main Data Analysis Script for AFT (Autonomous Flying Taxi) Survey Data
This script analyzes the raw survey data to understand data characteristics
and how the ML model might make predictions based on the main data.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_main_data(file_path):
    """
    Load and analyze the main survey data file
    
    Args:
        file_path (str): Path to the Excel file containing survey data
    
    Returns:
        pd.DataFrame: Processed data ready for ML model
    """
    
    print("="*80)
    print("MAIN DATA ANALYSIS FOR AFT SURVEY")
    print("="*80)
    
    # Load data
    print(f"\n1. LOADING DATA FROM: {file_path}")
    try:
        data = pd.read_excel(file_path)
        print(f"[OK] Data loaded successfully")
        print(f"  - Shape: {data.shape}")
        print(f"  - Columns: {len(data.columns)}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None
    
    # Initial data exploration
    print(f"\n2. INITIAL DATA EXPLORATION")
    print(f"   - Data types:")
    print(data.dtypes.value_counts())
    
    print(f"\n   - Column names:")
    for i, col in enumerate(data.columns, 1):
        print(f"     {i:2d}. {col}")
    
    # Check for missing values
    print(f"\n3. MISSING VALUES ANALYSIS")
    missing_data = data.isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    
    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_summary) > 0:
        print("   Missing values found:")
        print(missing_summary.to_string(index=False))
    else:
        print("   [OK] No missing values found")
    
    # Analyze CHOICE variable (target variable)
    print(f"\n4. TARGET VARIABLE ANALYSIS (CHOICE)")
    if 'CHOICE' in data.columns:
        print(f"   - CHOICE column found")
        print(f"   - Data type: {data['CHOICE'].dtype}")
        print(f"   - Unique values: {sorted(data['CHOICE'].unique())}")
        print(f"   - Value counts:")
        choice_counts = data['CHOICE'].value_counts().sort_index()
        for choice, count in choice_counts.items():
            percentage = (count / len(data)) * 100
            choice_name = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}.get(choice, f"Unknown ({choice})")
            print(f"     {choice} ({choice_name}): {count} ({percentage:.1f}%)")
    else:
        print("   [ERROR] CHOICE column not found")
    
    # Analyze key variables as described by user
    print(f"\n5. KEY VARIABLES ANALYSIS")
    
    # Likelihood variables (travel purpose)
    likelihood_cols = [col for col in data.columns if 'Likelihood' in col or 'likelihood' in col]
    if likelihood_cols:
        print(f"\n   a) LIKELIHOOD VARIABLES (Travel Purpose):")
        print(f"      Found columns: {likelihood_cols}")
        for col in likelihood_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Values: {sorted(data[col].unique())}")
                print(f"        Distribution: {data[col].value_counts().sort_index().to_dict()}")
    
    # Attitude towards autonomous technology
    attitude_cols = [col for col in data.columns if 'AtoLattitude' in col or 'attitude' in col.lower()]
    if attitude_cols:
        print(f"\n   b) ATTITUDE TOWARDS AUTONOMOUS TECHNOLOGY:")
        print(f"      Found columns: {attitude_cols}")
        for col in attitude_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Values: {sorted(data[col].unique())}")
                print(f"        Distribution: {data[col].value_counts().sort_index().to_dict()}")
    
    # Technology concern
    tech_concern_cols = [col for col in data.columns if 'technologyconcern' in col or 'technology' in col.lower()]
    if tech_concern_cols:
        print(f"\n   c) TECHNOLOGY CONCERN:")
        print(f"      Found columns: {tech_concern_cols}")
        for col in tech_concern_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Values: {sorted(data[col].unique())}")
                print(f"        Distribution: {data[col].value_counts().sort_index().to_dict()}")
    
    # Environmental concern
    env_concern_cols = [col for col in data.columns if 'environmentconcern' in col or 'environment' in col.lower()]
    if env_concern_cols:
        print(f"\n   d) ENVIRONMENTAL CONCERN:")
        print(f"      Found columns: {env_concern_cols}")
        for col in env_concern_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Values: {sorted(data[col].unique())}")
                print(f"        Distribution: {data[col].value_counts().sort_index().to_dict()}")
    
    # Cost variables
    cost_cols = [col for col in data.columns if 'co' in col.lower() or 'cost' in col.lower()]
    if cost_cols:
        print(f"\n   e) COST VARIABLES:")
        print(f"      Found columns: {cost_cols}")
        for col in cost_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Min: {data[col].min():.2f}")
                print(f"        Max: {data[col].max():.2f}")
                print(f"        Mean: {data[col].mean():.2f}")
                print(f"        Std: {data[col].std():.2f}")
    
    # Inconvenience variables
    inc_cols = [col for col in data.columns if 'inc' in col.lower() or 'inconvenience' in col.lower()]
    if inc_cols:
        print(f"\n   f) INCONVENIENCE VARIABLES:")
        print(f"      Found columns: {inc_cols}")
        for col in inc_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Min: {data[col].min():.2f}")
                print(f"        Max: {data[col].max():.2f}")
                print(f"        Mean: {data[col].mean():.2f}")
                print(f"        Std: {data[col].std():.2f}")
    
    # Multitasking variables
    multi_cols = [col for col in data.columns if 'multi' in col.lower() or 'multitasking' in col.lower()]
    if multi_cols:
        print(f"\n   g) MULTITASKING VARIABLES:")
        print(f"      Found columns: {multi_cols}")
        for col in multi_cols:
            if col in data.columns:
                print(f"      - {col}:")
                print(f"        Values: {sorted(data[col].unique())}")
                print(f"        Distribution: {data[col].value_counts().sort_index().to_dict()}")
    
    # Demographics analysis
    print(f"\n6. DEMOGRAPHICS ANALYSIS")
    demo_cols = [col for col in data.columns if any(x in col.lower() for x in ['age', 'gender', 'income', 'education', 'employment'])]
    if demo_cols:
        print(f"   Found demographic columns: {demo_cols}")
        for col in demo_cols:
            if col in data.columns:
                print(f"   - {col}:")
                if data[col].dtype in ['object', 'category']:
                    print(f"     Categories: {data[col].value_counts().to_dict()}")
                else:
                    print(f"     Min: {data[col].min()}, Max: {data[col].max()}, Mean: {data[col].mean():.2f}")
    
    # Data quality assessment
    print(f"\n7. DATA QUALITY ASSESSMENT")
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"   - Duplicate rows: {duplicates}")
    
    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        print(f"   - Constant columns (no variation): {constant_cols}")
    else:
        print(f"   - [OK] No constant columns found")
    
    # Check for high correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"   - High correlation pairs (>0.9):")
            for col1, col2, corr in high_corr_pairs:
                print(f"     {col1} - {col2}: {corr:.3f}")
        else:
            print(f"   - No high correlation pairs found")
    
    # Summary statistics
    print(f"\n8. SUMMARY STATISTICS")
    print(f"   - Total observations: {len(data)}")
    print(f"   - Total features: {len(data.columns)}")
    print(f"   - Numeric features: {len(data.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categorical features: {len(data.select_dtypes(include=['object', 'category']).columns)}")
    
    # Create visualizations
    print(f"\n9. CREATING VISUALIZATIONS")
    create_visualizations(data)
    
    # Save analysis results
    print(f"\n10. SAVING ANALYSIS RESULTS")
    save_analysis_results(data, file_path)
    
    return data

def create_visualizations(data):
    """Create visualizations for data analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path("../../../Result/DATA_ANALYSIS/Main_Data_Analysis_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Target variable distribution
    if 'CHOICE' in data.columns:
        plt.figure(figsize=(12, 6))
        choice_counts = data['CHOICE'].value_counts().sort_index()
        choice_labels = {1: "Car", 2: "Public Transport", 3: "Flying Taxi"}
        labels = [choice_labels.get(choice, f"Unknown ({choice})") for choice in choice_counts.index]
        
        plt.subplot(1, 2, 1)
        bars1 = choice_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Choice Distribution (Count)')
        plt.xlabel('Transport Mode')
        plt.ylabel('Count')
        plt.xticks(range(len(labels)), labels, rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(choice_counts.values):
            plt.text(i, v + 20, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.subplot(1, 2, 2)
        choice_percentages = (choice_counts / len(data)) * 100
        bars2 = choice_percentages.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Choice Distribution (Percentage)')
        plt.xlabel('Transport Mode')
        plt.ylabel('Percentage (%)')
        plt.xticks(range(len(labels)), labels, rotation=45)
        
        # Add percentage labels on bars
        for i, v in enumerate(choice_percentages.values):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'choice_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Environmental concern variables distribution
    env_concern_cols = [col for col in data.columns if 'environmentconcern' in col]
    if env_concern_cols:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Define labels and meanings for each environmental concern variable
        env_labels = {
            'environmentconcern_r1': {
                'title': 'Concerned about Global Warming',
                'meaning': '1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree',
                'description': 'How concerned are you about global warming?'
            },
            'environmentconcern_r2': {
                'title': 'Change Behavior for Environment',
                'meaning': '1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree, 6=Very Strongly Agree',
                'description': 'I do NOT change my behavior based on environmental concerns'
            },
            'environmentconcern_r3': {
                'title': 'Acceptable to Cause Pollution',
                'meaning': '1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree',
                'description': 'It is acceptable for society to cause some pollution'
            },
            'environmentconcern_r4': {
                'title': 'Pay More for Eco-Friendly Products',
                'meaning': '1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree',
                'description': 'I am willing to pay more for environmentally friendly products'
            }
        }
        
        for i, col in enumerate(env_concern_cols[:4]):
            if i < len(axes):
                # Get value counts and create bar plot
                value_counts = data[col].value_counts().sort_index()
                value_counts.plot(kind='bar', ax=axes[i], color='lightgreen', edgecolor='darkgreen')
                
                # Set title with meaning
                if col in env_labels:
                    title = f"{env_labels[col]['title']}\n({env_labels[col]['description']})"
                    axes[i].set_title(title, fontsize=10, pad=20)
                else:
                    axes[i].set_title(f'{col}', fontsize=10)
                
                axes[i].set_xlabel('Response Scale', fontsize=9)
                axes[i].set_ylabel('Count', fontsize=9)
                axes[i].tick_params(axis='x', rotation=0, labelsize=8)
                axes[i].tick_params(axis='y', labelsize=8)
                
                # Add value labels on bars
                for j, v in enumerate(value_counts.values):
                    axes[i].text(j, v + 10, str(v), ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(env_concern_cols), len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle('Environmental Concern Variables Distribution\n(Scale: 1=Strongly Disagree to 5/6=Strongly Agree)', 
                     fontsize=14, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(output_dir / 'environmental_concern_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Cost variables comparison
    cost_cols = [col for col in data.columns if 'co' in col.lower() and col != 'CHOICE']
    if cost_cols:
        plt.figure(figsize=(12, 8))
        
        # Box plot for cost comparison
        plt.subplot(2, 2, 1)
        cost_data = data[cost_cols]
        cost_data.boxplot()
        plt.title('Cost Variables Distribution')
        plt.ylabel('Cost')
        plt.xticks(rotation=45)
        
        # Correlation heatmap
        plt.subplot(2, 2, 2)
        corr_matrix = cost_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Cost Variables Correlation')
        
        # Mean cost by choice
        if 'CHOICE' in data.columns:
            plt.subplot(2, 2, 3)
            mean_costs = data.groupby('CHOICE')[cost_cols].mean()
            mean_costs.plot(kind='bar')
            plt.title('Mean Cost by Transport Choice')
            plt.xlabel('Transport Choice')
            plt.ylabel('Mean Cost')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Cost distribution
        plt.subplot(2, 2, 4)
        for col in cost_cols:
            plt.hist(data[col], alpha=0.7, label=col, bins=20)
        plt.title('Cost Variables Histogram')
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Demographics analysis
    demo_cols = [col for col in data.columns if any(x in col.lower() for x in ['age', 'gender', 'income', 'education'])]
    if demo_cols and 'CHOICE' in data.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(demo_cols[:4]):
            if i < len(axes) and col in data.columns:
                if data[col].dtype in ['object', 'category']:
                    # Categorical variable
                    cross_tab = pd.crosstab(data[col], data['CHOICE'])
                    bars = cross_tab.plot(kind='bar', ax=axes[i], stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    axes[i].set_title(f'{col} by Transport Choice', fontsize=12, pad=20)
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Count', fontsize=10)
                    axes[i].legend(['Car', 'Public Transport', 'Flying Taxi'], fontsize=9)
                    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
                    axes[i].tick_params(axis='y', labelsize=8)
                    
                    # Add value labels on stacked bars
                    for container in axes[i].containers:
                        axes[i].bar_label(container, fontsize=8, label_type='center')
                        
                else:
                    # Numeric variable - create histogram with value labels
                    choice_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
                    choice_names = {1: 'Car', 2: 'Public Transport', 3: 'Flying Taxi'}
                    
                    for choice in sorted(data['CHOICE'].unique()):
                        subset = data[data['CHOICE'] == choice][col]
                        if len(subset) > 0:
                            axes[i].hist(subset, alpha=0.7, label=f'{choice_names[choice]} (n={len(subset)})', 
                                       bins=20, color=choice_colors[choice])
                    
                    axes[i].set_title(f'{col} Distribution by Choice', fontsize=12, pad=20)
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Frequency', fontsize=10)
                    axes[i].legend(fontsize=9)
                    axes[i].tick_params(axis='x', labelsize=8)
                    axes[i].tick_params(axis='y', labelsize=8)
        
        # Hide unused subplots
        for i in range(len(demo_cols), len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle('Demographics Analysis by Transport Choice', fontsize=14, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(output_dir / 'demographics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   [OK] Visualizations saved to: {output_dir}")

def save_analysis_results(data, file_path):
    """Save analysis results to files"""
    
    # Create output directory
    output_dir = Path("../../../Result/DATA_ANALYSIS/Main_Data_Analysis_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data summary
    summary_file = output_dir / 'data_summary.txt'
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
        
        f.write("\nMISSING VALUES:\n")
        f.write("-"*30 + "\n")
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100
        for col in data.columns:
            if missing_data[col] > 0:
                f.write(f"{col}: {missing_data[col]} ({missing_percent[col]:.1f}%)\n")
        
        if missing_data.sum() == 0:
            f.write("No missing values found.\n")
    
    # Save descriptive statistics
    stats_file = output_dir / 'descriptive_statistics.csv'
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_stats = data[numeric_cols].describe()
        desc_stats.to_csv(stats_file)
    
    # Save correlation matrix
    if len(numeric_cols) > 1:
        corr_file = output_dir / 'correlation_matrix.csv'
        corr_matrix = data[numeric_cols].corr()
        corr_matrix.to_csv(corr_file)
    
    print(f"   [OK] Analysis results saved to: {output_dir}")

def main():
    """Main function to run the analysis"""
    
    # File path - Updated with correct location
    file_path = 'D:/Files_D/Study/==Thesis==/new_data/aft_2ndversion.xlsx'
    
    print("MAIN DATA ANALYSIS SCRIPT FOR AFT SURVEY")
    print("="*80)
    print(f"Target file: {file_path}")
    print("="*80)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"\n[WARNING] File not found at {file_path}")
        print("Please update the file_path variable in the script with the correct path.")
        print("\nTo use this script:")
        print("1. Update the file_path variable with your actual file location")
        print("2. Run the script: python Main_Data_Analysis_2.py")
        return
    
    # Run analysis
    try:
        data = load_and_analyze_main_data(file_path)
        if data is not None:
            print(f"\n[SUCCESS] Analysis completed successfully!")
            print(f"[SUCCESS] Results saved to: Result/DataPreprocessing_aft/Main_Data_Analysis_2/")
        else:
            print(f"\n[ERROR] Analysis failed!")
    except Exception as e:
        print(f"\n[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
