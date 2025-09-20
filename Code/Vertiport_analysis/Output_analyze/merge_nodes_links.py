import pandas as pd
import os

# File paths
nodes_file = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/nodes.csv'
links_file = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/links.csv'
output_file = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/links_with_coordinates.csv'

# Check if input files exist
if not os.path.exists(nodes_file):
    print(f"Error: Nodes file not found at: {nodes_file}")
    exit(1)

if not os.path.exists(links_file):
    print(f"Error: Links file not found at: {links_file}")
    exit(1)

try:
    print("Loading data...")
    # Load the nodes and links data into DataFrames
    nodes_df = pd.read_csv(nodes_file)
    links_df = pd.read_csv(links_file)
    
    print(f"Loaded {len(nodes_df)} nodes and {len(links_df)} links")
    
    # Display basic info about the data
    print(f"\nNodes columns: {list(nodes_df.columns)}")
    print(f"Links columns: {list(links_df.columns)}")
    
    # Check for any missing node IDs in the links
    from_nodes_in_links = set(links_df['from'].unique())
    to_nodes_in_links = set(links_df['to'].unique())
    all_nodes_in_links = from_nodes_in_links.union(to_nodes_in_links)
    nodes_in_nodes_df = set(nodes_df['id'].unique())
    
    missing_nodes = all_nodes_in_links - nodes_in_nodes_df
    if missing_nodes:
        print(f"\nWarning: {len(missing_nodes)} node IDs in links are not found in nodes file")
        print(f"First 10 missing nodes: {list(missing_nodes)[:10]}")
    else:
        print("\nAll node IDs in links are found in nodes file ✓")
    
    print("\nMerging data...")
    # Merge the 'from' node coordinates with the links DataFrame
    links_with_coords = links_df.merge(
        nodes_df[['id', 'x', 'y']], 
        left_on='from', 
        right_on='id', 
        suffixes=('', '_from_coords')
    )
    
    # Merge the 'to' node coordinates with the links DataFrame
    links_with_coords = links_with_coords.merge(
        nodes_df[['id', 'x', 'y']], 
        left_on='to', 
        right_on='id', 
        suffixes=('_from', '_to')
    )
    
    # Clean up the column names
    # Rename the coordinate columns to be more descriptive
    links_with_coords = links_with_coords.rename(columns={
        'x_from': 'from_x',
        'y_from': 'from_y',
        'x_to': 'to_x', 
        'y_to': 'to_y'
    })
    
    # Remove duplicate 'id' columns that were created during merge
    if 'id_from' in links_with_coords.columns:
        links_with_coords = links_with_coords.drop('id_from', axis=1)
    if 'id_to' in links_with_coords.columns:
        links_with_coords = links_with_coords.drop('id_to', axis=1)
    
    print(f"Successfully merged data. Final dataset has {len(links_with_coords)} links with coordinates")
    
    # Display sample of the merged data
    print(f"\nFinal columns: {list(links_with_coords.columns)}")
    print("\nSample of merged data:")
    sample_cols = ['id', 'from', 'to', 'from_x', 'from_y', 'to_x', 'to_y', 'length', 'type']
    available_cols = [col for col in sample_cols if col in links_with_coords.columns]
    print(links_with_coords[available_cols].head())
    
    # Save the merged data
    print(f"\nSaving merged data to: {output_file}")
    links_with_coords.to_csv(output_file, index=False)
    
    # Display summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total links processed: {len(links_with_coords)}")
    print(f"Links with valid coordinates: {len(links_with_coords.dropna(subset=['from_x', 'from_y', 'to_x', 'to_y']))}")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Calculate some basic statistics
    if 'length' in links_with_coords.columns:
        print(f"Average link length: {links_with_coords['length'].mean():.2f} meters")
        print(f"Total network length: {links_with_coords['length'].sum() / 1000:.2f} kilometers")
    
    if 'type' in links_with_coords.columns:
        print(f"\nRoad types distribution:")
        print(links_with_coords['type'].value_counts().head(10))
    
    print(f"\n✅ Successfully created links with coordinates!")
    print(f"📁 Output file: {output_file}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
