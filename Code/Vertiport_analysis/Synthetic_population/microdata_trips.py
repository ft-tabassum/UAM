import pandas as pd
import numpy as np

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
        child_count = len(group[group['age'] <= 17])

        # Categorize based on number of children
        if child_count == 0:
            child_household = 0  # No child or prefer not to answer
        elif child_count == 1:
            child_household = 1  # 1 child
        elif child_count == 2:
            child_household = 2  # 2 children
        else:
            child_household = 3  # 3 or more children

        adults_count = len(group[group['age'] >= 18])

        if adults_count == 0:
            adults_household = 0  # No adult or prefer not to answer
        elif adults_count == 1:
            adults_household = 1  # 1 adult
        elif adults_count == 2:
            adults_household = 2  # 2 adults
        elif adults_count == 3:
            adults_household = 3  # 3 adults
        else:
            adults_household = 4  # 4 or more adults

        return pd.Series({
            'child_household': child_household,
            'adults_household': adults_household
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
    If any value is exactly 166.66666666666666, replace it with 99999.
    """
    pt_times = []
    for col in ['time_bus', 'time_train', 'time_tram_metro']: #time is in min
        time_val = row.get(col, np.inf)  # Get time value for the column, default to infinity if not found
        if not pd.isna(time_val):  # Ensure the value is not NaN
            if time_val == 166.66666666666666:
                time_val = 99999
            pt_times.append(time_val)  # Add the time value to the list

    # Return the smallest time among the available ones
    return min(pt_times) if pt_times else np.nan


# Function to calculate travel costs for auto and public transport (PT)
def calculate_travel_costs(distance, time_pt):
    """ Calculate travel costs based on distance and time. #distance is in "km" """

    # Cost parameters
    circuity_factor = 1.215 # (Kim et al., 2025)
    cost_per_km_auto = 0.65  #unit: €/km (Manuscript Number: JTRP-D-24-00632R1)
    base_fare_pt = 4.10 #MVV- single trip ticket
    average_cost_per_km_pt = 0.26 # region trip-longer trip (Schröder & Gotzler, 2021)

    # Auto cost calculation
    travel_cost_auto = distance * circuity_factor * cost_per_km_auto

    # PT cost calculation
    travel_cost_pt = base_fare_pt + (distance * circuity_factor* average_cost_per_km_pt)  if not pd.isna(time_pt) else np.nan

    return travel_cost_auto, travel_cost_pt


# Main function to process and combine all data with calculations
def process_combined_data():
    print("Starting combined data processing...")
    print("=" * 60)

    # Define file paths
    file_paths = {
        'hh': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\hh_2011.csv",
        'pp': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\sp\pp_2011.csv",
        'trips': r"D:\Files_D\Study\Thesis\data\travel_demand_2021\travel_demand_2021\trips\trips.csv"
    }

    try:
        # Read all data files
        print("Reading data files...")
        hh_data = pd.read_csv(file_paths['hh'])
        pp_data = pd.read_csv(file_paths['pp'])
        trips_data = pd.read_csv(file_paths['trips'])

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

        # Merge with trips data (INNER JOIN - only people with trips)
        print("Merging with trips data (INNER JOIN - only people with trips)...")
        combined_data = combined_data.merge(
            trips_data[['trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY', 'id', 'distance', 'time_auto', 'time_bus', 'time_train', 'time_tram_metro', 'purpose']],
            left_on='id',
            right_on='id',
            how='inner',  # Changed from 'left' to 'inner'
            suffixes=('', '_trips')
        )
        #Filtering trips with distance greater than or equal 20 km
        print("Filtering trips with distance >= 20 km...")
        combined_data = combined_data[combined_data['distance'] >= 20]

        # Calculate the shortest PT time
        print("Calculating shortest PT time...")
        combined_data['time_PT'] = combined_data.apply(lambda row: select_shortest_pt_time(row), axis=1)

        # Calculate travel costs
        print("Calculating travel costs...")
        costs = combined_data.apply(
            lambda row: calculate_travel_costs(row['distance'], row['time_PT']),
            axis=1 )
        combined_data['travel_cost_auto'] = costs.apply(lambda x: x[0])
        combined_data['travel_cost_pt'] = costs.apply(lambda x: x[1])

        # Separate travel times into in-vehicle and waiting times

        # For auto
        combined_data['in_vehicle_time_auto'] = combined_data['time_auto']  # In-vehicle time is same as time_auto
        combined_data['waiting_time_auto'] = 0  # No waiting time for auto
        # Rename time_auto to travel_time_auto
        combined_data.rename(columns={'time_auto': 'travel_time_auto'}, inplace=True)

        # For public transport
        combined_data['waiting_time_pt'] = np.where(combined_data['time_PT'] > 100, 20,5)  # Waiting time is 20 if time_pt > 100, else 5
        combined_data['in_vehicle_time_pt'] = combined_data['time_PT'] - combined_data['waiting_time_pt']  # In-vehicle time = time_pt - waiting_time
        # Rename time_pt to travel_time_pt
        combined_data.rename(columns={'time_PT': 'travel_time_pt'}, inplace=True)

        print("Travel times separated and renamed successfully.")

        # Rename columns for trips data (id -> person_id, trip_id is already in trips data)
        combined_data.rename(columns={'id': 'person_id'}, inplace=True)

        # Select and reorder final columns
        final_columns = [
            'trip_id', 'origin', 'originX', 'originY', 'destination', 'destinationX', 'destinationY',
            'person_id', 'age', 'gender', 'child_Household', 'occupation', 'adult_household', 'driversLicense',
            'income', 'disability', 'purpose', 'autos', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
            'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto', 'travel_cost_pt'
        ]

        # Filter to only include columns that exist
        available_columns = [col for col in final_columns if col in combined_data.columns]
        final_data = combined_data[available_columns]

        # Save the combined data
        output_file = "../../../Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/microdata_trips.csv"
        final_data.to_csv(output_file, index=False)
        return final_data

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the data files. {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None


# Main function to execute the combined data processing
def main():
    print("Starting Combined Data Processing and Calculations")
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