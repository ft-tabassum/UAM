import pandas as pd
import numpy as np
import os

# Function to calculate Child_Household and Adult_Household based on age
def calculate_household_composition(pp_data, hh_data):
    """
    Calculate Child_Household and Adult_household based on hhid and age.
    Children: age <= 17, Adults: age >= 18
    """
    # Merge pp and hh data on hhid
    merged_data = pp_data.merge(hh_data[['id', 'hhSize']], left_on='hhid', right_on='id', how='left',
                                suffixes=('', '_hh'))

    # Calculate household composition
    def calculate_household_stats(group):
        children = len(group[group['age'] <= 17])
        adults = len(group[group['age'] >= 18])
        return pd.Series({
            'Child_Household': children,
            'Adult_household': adults
        })

    # Group by hhid and calculate household composition
    household_stats = merged_data.groupby('hhid').apply(calculate_household_stats).reset_index()

    # Merge back to pp_data
    result = pp_data.merge(household_stats, on='hhid', how='left')

    return result


# Function to select the shortest public transport time
def select_shortest_pt_time(row):
    """
    Select the shortest time among time_bus, time_train, time_tram_metro.
    All values, including dummy values, are considered.
    """
    pt_times = []
    for col in ['time_bus', 'time_train', 'time_tram_metro']:
        time_val = row.get(col, np.inf)  # Get time value for the column, default to infinity if not found
        if not pd.isna(time_val):  # Ensure the value is not NaN
            pt_times.append(time_val)  # Add the time value to the list

    # Return the smallest time among the available ones
    return min(pt_times) if pt_times else np.nan


# Function to calculate travel costs for auto and public transport (PT)
def calculate_travel_costs(distance, time_auto, time_pt):
    """
    Calculate travel costs based on distance and time.
    NOTE: This function uses placeholder values. You should replace these with actual cost parameters
    based on your specific study requirements or data from your files.
    """
    # Placeholder cost parameters (replace with actual values)
    cost_per_km_auto = 0.15  # Replace with actual auto cost per km
    cost_per_minute_auto = 0.10  # Replace with actual auto cost per minute
    cost_per_km_pt = 0.05  # Replace with actual PT cost per km
    cost_per_minute_pt = 0.05  # Replace with actual PT cost per minute

    # Auto cost calculation
    TravelCost_auto = (distance * cost_per_km_auto) + (time_auto * cost_per_minute_auto)

    # PT cost calculation
    TravelCost_PT = (distance * cost_per_km_pt) + (time_pt * cost_per_minute_pt) if not pd.isna(time_pt) else np.nan

    return TravelCost_auto, TravelCost_PT


# Main function to process and combine all data with calculations
def process_combined_data():
    print("Starting combined data processing...")
    print("=" * 60)

    # Define file paths
    file_paths = {
        'dd': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\dd_2011.csv",
        'ee': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\ee_2011.csv",
        'hh': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\hh_2011.csv",
        'jj': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\jj_2011.csv",
        'pp': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\pp_2011.csv",
        'trips': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\trips\trips.csv"
    }

    try:
        # Read all data files
        print("Reading data files...")
        dd_data = pd.read_csv(file_paths['dd'])
        ee_data = pd.read_csv(file_paths['ee'])
        hh_data = pd.read_csv(file_paths['hh'])
        jj_data = pd.read_csv(file_paths['jj'])
        pp_data = pd.read_csv(file_paths['pp'])
        trips_data = pd.read_csv(file_paths['trips'])

        print(f"Successfully read all data files:")
        print(f"  - dd: {len(dd_data)} rows")
        print(f"  - ee: {len(ee_data)} rows")
        print(f"  - hh: {len(hh_data)} rows")
        print(f"  - jj: {len(jj_data)} rows")
        print(f"  - pp: {len(pp_data)} rows")
        print(f"  - trips: {len(trips_data)} rows")
        print()

        # FILTER OUT BICYCLE AND WALK MODES
        trips_data = trips_data[~trips_data['mode'].isin(['bicycle', 'walk'])]
        print(f"Filtered trips: {len(trips_data)} rows remain after excluding 'bicycle' and 'walk' modes.")

        # Calculate household composition
        print("Calculating household composition...")
        pp_with_household = calculate_household_composition(pp_data, hh_data)

        # Start with pp data as base
        combined_data = pp_with_household.copy()

        # Add required columns from other datasets
        print("Adding columns from other datasets...")

        # Add autos from hh data
        combined_data = combined_data.merge(
            hh_data[['id', 'autos']],
            left_on='hhid',
            right_on='id',
            how='left',
            suffixes=('', '_hh')
        )

        # FIXED: Use INNER JOIN instead of LEFT JOIN to only include people with trips
        print("Merging with trips data (INNER JOIN - only people with trips)...")
        combined_data = combined_data.merge(
            trips_data[['trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY', 'id', 'distance', 'time_auto', 'time_bus', 'time_train', 'time_tram_metro', 'purpose']],
            left_on='id',
            right_on='id',
            how='inner',  # Changed from 'left' to 'inner'
            suffixes=('', '_trips')
        )

        # Calculate shortest PT time
        print("Calculating shortest PT time...")
        combined_data['time_PT'] = combined_data.apply(lambda row: select_shortest_pt_time(row), axis=1)

        # Calculate travel costs
        print("Calculating travel costs...")
        costs = combined_data.apply(
            lambda row: calculate_travel_costs(row['distance'], row['time_auto'], row['time_PT']),
            axis=1
        )
        combined_data['TravelCost_auto'] = costs.apply(lambda x: x[0])
        combined_data['TravelCost_PT'] = costs.apply(lambda x: x[1])

        # Rename columns for trips data (id -> person_id, trip_id is already in trips data)
        combined_data.rename(columns={'id': 'person_id'}, inplace=True)

        # Select and reorder final columns
        final_columns = [
            'trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY',
            'person_id', 'age', 'gender', 'Child_Household', 'occupation', 'Adult_household', 'driversLicense',
            'income', 'education', 'disability', 'purpose', 'autos', 'distance', 'time_auto', 'time_PT',
            'TravelCost_auto', 'TravelCost_PT'
        ]

        # Filter to only include columns that exist
        available_columns = [col for col in final_columns if col in combined_data.columns]
        final_data = combined_data[available_columns]

        # Create output directory
        output_dir = "../../../Result/Vertiport_analysis/Model_XgBoost/Synthetic_population"
        os.makedirs(output_dir, exist_ok=True)

        # Save the combined data
        output_file = f"{output_dir}/trial_micro_trips_no_bike_walk.csv"
        final_data.to_csv(output_file, index=False)

        print(f"Successfully created combined dataset:")
        print(f"  - Output file: {output_file}")
        print(f"  - Total rows: {len(final_data)}")
        print(f"  - Columns: {list(final_data.columns)}")
        print()

        # Display some statistics
        print("Data Summary:")
        print(f"  - Average age: {final_data['age'].mean():.2f}")
        print(f"  - Average household size (children): {final_data['Child_Household'].mean():.2f}")
        print(f"  - Average household size (adults): {final_data['Adult_household'].mean():.2f}")
        print(f"  - Average distance: {final_data['distance'].mean():.2f} km")
        print(f"  - Average auto cost: ${final_data['TravelCost_auto'].mean():.2f}")
        print(f"  - Average PT cost: ${final_data['TravelCost_PT'].mean():.2f}")

        return final_data

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the data files. {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None


# Main function to execute the combined data processing
def main():
    print("Starting Combined Data Processing and Calculations (NO BIKE/WALK VERSION)")
    print("=" * 60)

    result = process_combined_data()

    if result is not None:
        print("=" * 60)
        print("Combined data processing completed successfully!")
    else:
        print("=" * 60)
        print("Combined data processing failed!")


if __name__ == "__main__":
    main() 