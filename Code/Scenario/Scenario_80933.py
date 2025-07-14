import pandas as pd

# Load the two sheets into DataFrames
df1 = pd.read_excel('D:/Files_D/Study/Thesis/data_ANALYSIS/80933_zone_attribute.xlsx')
df2 = pd.read_excel('D:/Files_D/Study/Thesis/data_ANALYSIS/80933_zone_TD.xlsx')

# Merge the two DataFrames on 'ZoneId'
merged_df = pd.merge(df1, df2, on='ZoneId', how='inner')

# Display the merged DataFrame
print("\nMerged DataFrame columns:", merged_df.columns.tolist())
print(merged_df.head())

# Save the merged DataFrame to a CSV file
# Reorder columns: keep 'X' and 'Y' beside 'centroidX' and 'centroidY' if they exist
cols = merged_df.columns.tolist()
if 'centroidX' in cols and 'centroidY' in cols and 'X' in cols and 'Y' in cols:
    # Remove 'X' and 'Y' from their current positions
    cols.remove('X')
    cols.remove('Y')
    # Find the position after 'centroidY'
    idx = cols.index('centroidY') + 1
    # Insert 'X' and 'Y' after 'centroidY'
    cols[idx:idx] = ['X', 'Y']
    merged_df = merged_df[cols]

merged_df.to_csv('D:/Thesis/UAM/Result/Scenario/Scenario_80933_zone.csv', index=False)
print("Merged DataFrame saved as '80933_zone.csv'.")