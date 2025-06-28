import pandas as pd

# Load files
files = {
    'trips': 'D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/trips/trips.csv',  # main trips file
    'pp': 'D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/sp/pp_2011.csv',
    'hh': 'D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/sp/hh_2011.csv', 
    'jj': 'D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/sp/jj_2011.csv',
    'ee': 'D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/sp/ee_2011.csv',
    'dd': 'D:/Files_D/Study/Thesis/data/travel_demand_2021/travel_demand_2021/sp/dd_2011.csv'
}

print("Starting data merging process...")

# Step 1: Read and merge pp, hh, jj, ee, dd files based on 'id' column
print("Step 1: Reading and merging pp, hh, jj, ee, dd files...")

# Read each file with specific columns
print("Reading pp file...")
pp = pd.read_csv(files['pp'], usecols=['id', 'age', 'gender', 'occupation', 'driversLicense', 'income'])
pp = pp.rename(columns={'occupation': 'employment', 'driversLicense': 'driving license'})

print("Reading hh file...")
hh = pd.read_csv(files['hh'], usecols=['id', 'zone', 'autos'])
hh = hh.rename(columns={'zone': 'household_zone', 'autos': 'household car'})

print("Reading jj file...")
jj = pd.read_csv(files['jj'], usecols=['id', 'zone', 'coordX', 'coordY'])
jj = jj.rename(columns={'zone': 'job_zone', 'coordX': 'coordx_job', 'coordY': 'coordy_job'})

print("Reading ee file...")
ee = pd.read_csv(files['ee'], usecols=['id', 'zone', 'coordX', 'coordY'])
ee = ee.rename(columns={'zone': 'school_zone', 'coordX': 'coordx_sch', 'coordY': 'coordy_sch'})

print("Reading dd file...")
dd = pd.read_csv(files['dd'], usecols=['id', 'coordX', 'coordY'])
dd = dd.rename(columns={'coordX': 'coordx_hh', 'coordY': 'coordy_hh'})

print(f"  - pp file: {len(pp)} rows, {len(pp.columns)} columns")
print(f"  - hh file: {len(hh)} rows, {len(hh.columns)} columns")
print(f"  - jj file: {len(jj)} rows, {len(jj.columns)} columns")
print(f"  - ee file: {len(ee)} rows, {len(ee.columns)} columns")
print(f"  - dd file: {len(dd)} rows, {len(dd.columns)} columns")

# Merge all files on 'id' column
merged = dd
for df_name, df in [('ee', ee), ('hh', hh), ('jj', jj), ('pp', pp)]:
    print(f"  - Merging {df_name}...")
    merged = pd.merge(merged, df, on='id', how='inner')
    print(f"    After merging {df_name}: {len(merged)} rows")

print(f"Step 1 complete. Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")

# Step 2: Read trips file and merge with the result from Step 1
print("\nStep 2: Reading trips file and merging...")

# Read trips file
trips = pd.read_csv(files['trips'])
print(f"  - trips file: {len(trips)} rows, {len(trips.columns)} columns")

# Rename the 'id' column in trips to 'trip_id' to avoid conflicts
trips = trips.rename(columns={'id': 'trip_id'})

# Merge trips with the merged dataset
# Note: trips uses 'person' column, merged uses 'id' column
final_merged = pd.merge(trips, merged, left_on='person', right_on='id', how='inner')

# Rename 'person' to 'person_id' for clarity
final_merged = final_merged.rename(columns={'person': 'person_id'})

# Drop the 'id' column from the final output since we have person_id
if 'id' in final_merged.columns:
    final_merged = final_merged.drop(columns=['id'])

print(f"Step 2 complete. Final merged dataset: {len(final_merged)} rows, {len(final_merged.columns)} columns")

# Save the final merged dataset
output_file = '../../Result/Data_Preprocessing/Synthetic_tripData.csv'
final_merged.to_csv(output_file, index=False)

print(f"\n✅ Merging completed successfully!")
print(f"📁 Output saved as: {output_file}")
print(f"📊 Final dataset: {len(final_merged)} rows, {len(final_merged.columns)} columns")

# Display column names for verification
print(f"\n📋 Column names in final dataset:")
for i, col in enumerate(final_merged.columns, 1):
    print(f"  {i:2d}. {col}") 