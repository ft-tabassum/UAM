import pandas as pd
import numpy as np

def filter_moosach_trips():
    """
    Filter trips to only include those related to Moosach zones
    Uses Scenario_80933_zone.csv as a lookup to identify Moosach zones
    """
    
    print("=== FILTERING TRIPS FOR MOOSACH ANALYSIS ===")
    
    # Read the files
    print("Reading files...")
    zones_df = pd.read_csv("Result/Scenario/Scenario_80933_zone.csv")
    trips_df = pd.read_csv("Result/Scenario/matching_origin_destination.csv")
    
    print(f"Zones file shape: {zones_df.shape}")
    print(f"Trips file shape: {trips_df.shape}")
    
    # Get Moosach zone IDs
    moosach_zones = zones_df['ZoneId'].unique()
    print(f"\nMoosach zones found: {len(moosach_zones)}")
    print(f"Moosach zone IDs: {sorted(moosach_zones)}")
    
    # Filter trips where either origin OR destination is in Moosach
    print("\nFiltering trips related to Moosach...")
    moosach_related_trips = trips_df[
        trips_df['origin'].isin(moosach_zones) |
        trips_df['destination'].isin(moosach_zones)
    ]
    
    print(f"Total trips: {len(trips_df)}")
    print(f"Moosach-related trips: {len(moosach_related_trips)}")
    print(f"Percentage of trips related to Moosach: {len(moosach_related_trips)/len(trips_df)*100:.1f}%")
    
    # Analyze the filtered data
    print(f"\n=== MOOSACH TRIPS ANALYSIS ===")
    
    # Check origin vs destination involvement
    origin_in_moosach = moosach_related_trips['origin'].isin(moosach_zones).sum()
    dest_in_moosach = moosach_related_trips['destination'].isin(moosach_zones).sum()
    both_in_moosach = moosach_related_trips[
        moosach_related_trips['origin'].isin(moosach_zones) & 
        moosach_related_trips['destination'].isin(moosach_zones)
    ].shape[0]
    
    print(f"Trips with origin in Moosach: {origin_in_moosach}")
    print(f"Trips with destination in Moosach: {dest_in_moosach}")
    print(f"Trips with both origin and destination in Moosach: {both_in_moosach}")
    
    # Show unique zones involved
    unique_origins = moosach_related_trips['origin'].unique()
    unique_destinations = moosach_related_trips['destination'].unique()
    
    print(f"\nUnique origin zones in Moosach trips: {len(unique_origins)}")
    print(f"Unique destination zones in Moosach trips: {len(unique_destinations)}")
    
    # Show which Moosach zones are most active
    print(f"\n=== MOOSACH ZONE ACTIVITY ===")
    origin_counts = moosach_related_trips['origin'].value_counts()
    dest_counts = moosach_related_trips['destination'].value_counts()
    
    print("Most active Moosach zones as ORIGIN:")
    for zone in moosach_zones:
        if zone in origin_counts.index:
            print(f"Zone {zone}: {origin_counts[zone]} trips")
    
    print(f"\nMost active Moosach zones as DESTINATION:")
    for zone in moosach_zones:
        if zone in dest_counts.index:
            print(f"Zone {zone}: {dest_counts[zone]} trips")
    
    # Show sample of filtered data
    print(f"\n=== SAMPLE MOOSACH TRIPS ===")
    sample_columns = ['trip_id', 'person_id', 'origin', 'destination', 'purpose', 'tripLength-km']
    print(moosach_related_trips[sample_columns].head(10))
    
    # Save the filtered data
    output_path = "Result/Scenario/moosach_related_trips.csv"
    moosach_related_trips.to_csv(output_path, index=False)
    print(f"\nFiltered Moosach trips saved to: {output_path}")
    
    # Create summary statistics
    summary_stats = {
        'total_trips': len(trips_df),
        'moosach_related_trips': len(moosach_related_trips),
        'percentage_moosach_trips': len(moosach_related_trips)/len(trips_df)*100,
        'moosach_zones_count': len(moosach_zones),
        'trips_origin_moosach': origin_in_moosach,
        'trips_dest_moosach': dest_in_moosach,
        'trips_both_moosach': both_in_moosach,
        'unique_origins_in_trips': len(unique_origins),
        'unique_destinations_in_trips': len(unique_destinations)
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv("Result/Scenario/moosach_filter_summary.csv", index=False)
    print(f"Summary statistics saved to: Result/Scenario/moosach_filter_summary.csv")
    
    return moosach_related_trips, summary_stats

if __name__ == "__main__":
    filtered_trips, stats = filter_moosach_trips() 