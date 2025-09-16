import pandas as pd
import sys
from datetime import datetime

# Create output file
output_file = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/rtts_statistics.txt'

# Function to print and save to file
def print_and_save(text, file_handle):
    print(text)
    file_handle.write(text + '\n')

# Load the data
data = pd.read_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

# Open file for writing
with open(output_file, 'w', encoding='utf-8') as f:
    print_and_save(f"RTTs Statistics Analysis - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
    print_and_save("=" * 80, f)

    # Calculate rtts for UAM
    data['rtts'] = 1 - (data['travel_time_Uam'] / data['autos_TT'])

    # Count trips with positive rtts and rtts > 0.5
    total_trips = len(data)
    positive_rtts_count = len(data[data['rtts'] > 0])
    high_rtts_count = len(data[data['rtts'] > 0.5])

    # Calculate average rtts for all trips
    average_rtts_all = data['rtts'].mean()
    average_rtts_positive = data[data['rtts'] > 0]['rtts'].mean()

    print_and_save("=" * 60, f)
    print_and_save("RTTs STATISTICS SUMMARY", f)
    print_and_save("=" * 60, f)
    print_and_save(f"Total trips in dataset: {total_trips:,}", f)
    print_and_save(f"Trips with positive rtts: {positive_rtts_count:,} ({positive_rtts_count/total_trips*100:.1f}%)", f)
    print_and_save(f"Trips with rtts > 0.5: {high_rtts_count:,} ({high_rtts_count/total_trips*100:.1f}%)", f)
    print_and_save(f"Average rtts for ALL trips: {average_rtts_all:.4f}", f)
    print_and_save(f"Average rtts for positive rtts trips: ({average_rtts_positive*100:.2f}%)", f)
    print_and_save("=" * 60, f)

    # Find trip length threshold where rtts becomes positive
    print_and_save("\nTrip Length Analysis for Positive RTTs:", f)
    positive_data = data[data['rtts'] > 0].copy()
    if len(positive_data) > 0:
        min_positive_trip_length = positive_data['trip_length'].min()
        max_positive_trip_length = positive_data['trip_length'].max()
        avg_positive_trip_length = positive_data['trip_length'].mean()
        
        print_and_save(f"Minimum trip length with positive rtts: {min_positive_trip_length/1000:.1f} km", f)
        print_and_save(f"Maximum trip length with positive rtts: {max_positive_trip_length/1000:.1f} km", f)
        print_and_save(f"Average trip length with positive rtts: {avg_positive_trip_length/1000:.1f} km", f)
        
        # Find the threshold more precisely by looking at trip length bins
        print_and_save(f"\nDetailed analysis by trip length bins:", f)
        trip_length_bins = pd.cut(data['trip_length'], bins=range(0, int(data['trip_length'].max()) + 5000, 5000))
        binned_analysis = data.groupby(trip_length_bins).agg({
            'rtts': ['mean', 'count'],
            'trip_length': 'mean'
        }).reset_index()
        
        # Flatten column names
        binned_analysis.columns = ['trip_length_bin', 'rtts_mean', 'count', 'trip_length_mean']
        
        # Find first bin with positive mean rtts
        positive_bins = binned_analysis[binned_analysis['rtts_mean'] > 0]
        if len(positive_bins) > 0:
            first_positive_bin = positive_bins.iloc[0]
            print_and_save(f"First trip length bin with positive average rtts: {first_positive_bin['trip_length_mean']/1000:.1f} km", f)
            print_and_save(f"Average rtts in this bin: {first_positive_bin['rtts_mean']:.4f}", f)
            print_and_save(f"Number of trips in this bin: {first_positive_bin['count']:,}", f)
    else:
        print_and_save("No positive rtts found in the dataset", f)

    print_and_save("=" * 60, f)

    # Additional statistics
    print_and_save("\nAdditional Statistics:", f)
    print_and_save(f"Minimum rtts: {data['rtts'].min():.4f}", f)
    print_and_save(f"Maximum rtts: {data['rtts'].max():.4f}", f)
    print_and_save(f"Median rtts: {data['rtts'].median():.4f}", f)
    print_and_save(f"Standard deviation rtts: {data['rtts'].std():.4f}", f)
    print_and_save("=" * 60, f)
    
    print_and_save(f"\nResults saved to: {output_file}", f)
