import pandas as pd
from collections import Counter

# Step 1: Read the file (assuming the file is a CSV)
file_path = "D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/trips/trips.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Step 2: Filter the rows where any of the values in 'time_bus', 'time_train', 'time_tram_metro' columns are greater than 100
# We use '&' to filter across the columns individually, ensuring we correctly capture values greater than 100
filtered_df = df[(df['time_bus'] > 100) | (df['time_train'] > 100) | (df['time_tram_metro'] > 100)]

# Step 3: Extract the filtered values from all the columns
filtered_values = pd.concat([filtered_df['time_bus'], filtered_df['time_train'], filtered_df['time_tram_metro']], ignore_index=True)

# Step 4: Filter out NaN values and count the frequency of each value greater than 100
filtered_values = filtered_values[filtered_values > 100]

# Step 5: Count the frequency of each value
value_counts = Counter(filtered_values)

# Step 6: Print the result
#print("Filtered values greater than 100 and their counts:")
#for value, count in value_counts.items():
 #   print(f"Value: {value}, Count: {count}")

#Step 6: Create a DataFrame from the counts
count_df = pd.DataFrame(value_counts.items(), columns=["Value", "Count"])

# Step 7: Save the result to a CSV file
output_file_path = "/Result/ignor/Model_XgBoost/Synthetic_population/tt_check.csv"  # Update with the desired output file path
count_df.to_csv(output_file_path, index=False)

# Print a message to confirm the file is saved
print(f"Result saved as CSV to: {output_file_path}")
