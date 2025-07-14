import pandas as pd

# Load the Excel file, this file is the updated version of matching_origin_destination.csv
#only the zoneId: [1734, 1736, 1737, 1739, 1740, 1746, 1748, 1751, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1765,1766]- are considered, representing 80933 PLZ.

file_path = "D:/Files_D/Study/Thesis/data_ANALYSIS/80933_same_OD_Zone.xlsx"
# List all sheet names
xls = pd.ExcelFile(file_path)
print("Sheet names:", xls.sheet_names)

# Read only the 'Destination' sheet
destination_df = pd.read_excel(file_path, sheet_name='Destination')
print("Destination sheet shape:", destination_df.shape)
print("Destination sheet columns:", list(destination_df.columns))
print("\nDestination sheet preview:")
print(destination_df.head())

# Save the destination sheet as a separate CSV file
output_path = "D:/Thesis/UAM/Result/Scenario/destination_Zone.csv"
destination_df.to_csv(output_path, index=False)
print(f"\nDestination data saved to: {output_path}")