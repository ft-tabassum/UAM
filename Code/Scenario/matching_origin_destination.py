import pandas as pd

# List of values to check
values_to_check = [
    1734, 1736, 1737, 1739, 1740, 1746, 1748, 1751, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1765,1766]

# Path to the CSV file
csv_path = 'D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/synthetic_population_processing.csv'  # Update path if needed
output_csv = 'D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/matching_origin_destination.csv'

# Read the CSV file
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"File not found: {csv_path}")
    exit(1)

# Check if required columns exist
if 'origin' not in df.columns or 'destination' not in df.columns:
    print("CSV must contain 'origin' and 'destination' columns.")
    exit(1)

# Find rows where origin or destination is in the list
mask = df['origin'].isin(values_to_check) | df['destination'].isin(values_to_check)
matching_rows = df[mask]

# Print the matching rows
if not matching_rows.empty:
    print("Rows with specified origin or destination values:")
    print(matching_rows)
    # Save to CSV
    matching_rows.to_csv(output_csv, index=False)
    print(f"Matching rows saved to {output_csv}")
else:
    print("No rows found with the specified origin or destination values.") 