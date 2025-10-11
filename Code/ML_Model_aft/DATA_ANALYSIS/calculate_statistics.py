import pandas as pd
import numpy as np

print("STATISTICS FOR AFT PROCESSED DATA")
print("=" * 50)

# Load the AFT processed data
data = pd.read_csv('/Result/DataPreprocessing_aft/aft_processed.csv')

print(f"Dataset loaded: {len(data):,} observations")
print(f"Total variables: {len(data.columns)}")

# Define the variables to analyze
variables = [
    'CAR_TT', 'PT_TT', 'AFT_TT',
    'CAR_CO', 'PT_CO', 'AFT_CO', 
    'CAR_SAFETY_ds', 'PT_SAFETY_safer', 'AFT_SAFETY_ds', 'AFT_SAFETY_riskier','AFT_SAFETY_safer',
    'CAR_INC', 'PT_INC', 'AFT_INC',
    'CAR_MULTI_no', 'PT_MULTI_inpart', 'AFT_MULTI_yes'
]

print(f"\nVARIABLES TO ANALYZE:")
for var in variables:
    print(f"  {var}")

# Check which variables exist in the dataset
print(f"\nCHECKING VARIABLE AVAILABILITY:")
available_vars = []
missing_vars = []

for var in variables:
    if var in data.columns:
        available_vars.append(var)
        print(f"  {var}: Available")
    else:
        missing_vars.append(var)
        print(f"  {var}: Not found")

# Calculate statistics for available variables
if available_vars:
    print(f"\nSTATISTICS FOR AVAILABLE VARIABLES:")
    print("=" * 60)
    print(f"{'Variable':<12} {'Count':<8} {'Mean':<8} {'Std Dev':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)
    
    results = []
    
    for var in available_vars:
        values = data[var].dropna()
        if len(values) > 0:
            count = len(values)
            mean_val = values.mean()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            
            print(f"{var:<12} {count:<8} {mean_val:<8.3f} {std_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f}")
            
            results.append({
                'Variable': var,
                'Count': count,
                'Mean': mean_val,
                'Std_Dev': std_val,
                'Min': min_val,
                'Max': max_val
            })
        else:
            print(f"{var:<12} 0       0.000    0.000    0.000    0.000")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.round(3)
        results_df.to_csv('D:/Thesis/UAM/Result/ML_models_aft/Variable_Statistics.csv', index=False)
        print(f"\nResults saved to: Variable_Statistics.csv")

# Calculate percentages for binary variables
print(f"\nPERCENTAGES FOR BINARY VARIABLES:")
print("=" * 40)

binary_vars = ['Commuting', 'Noncommuting', 'adult']
for var in binary_vars:
    if var in data.columns:
        values = data[var].dropna()
        if len(values) > 0:
            # For binary variables, calculate percentage of 1s
            percentage_1 = (values.sum() / len(values)) * 100
            percentage_0 = 100 - percentage_1
            print(f"{var}:")
            print(f"  1 (Yes): {values.sum():,} ({percentage_1:.1f}%)")
            print(f"  0 (No): {len(values) - values.sum():,} ({percentage_0:.1f}%)")
        else:
            print(f"{var}: No data available")
    else:
        print(f"{var}: Variable not found")

# Check for missing variables
if missing_vars:
    print(f"\nMISSING VARIABLES:")
    print("=" * 20)
    for var in missing_vars:
        print(f"  {var}")
    
    # Check for similar variable names
    print(f"\nCHECKING FOR SIMILAR VARIABLE NAMES:")
    print("=" * 40)
    for var in missing_vars:
        # Look for variables with similar names
        similar_vars = [col for col in data.columns if var.split('_')[0] in col and var.split('_')[1] in col]
        if similar_vars:
            print(f"  {var} -> Similar: {similar_vars}")
        else:
            print(f"  {var} -> No similar variables found")

# Show all available variables for reference
print(f"\nALL AVAILABLE VARIABLES IN DATASET:")
print("=" * 40)
all_vars = sorted(data.columns.tolist())
for i, var in enumerate(all_vars):
    if i % 4 == 0:
        print()
    print(f"{var:<20}", end="")
print()

print("\n" + "="*50)
print("STATISTICS CALCULATION COMPLETE!")
print("="*50)
