"""
Convert Lat/Long (EPSG:4326) to EPSG:31468 (Gauss-Kruger Zone 4)
Simple script to convert 5 places from internet coordinates
"""

from pyproj import Transformer

# Create transformer from WGS84 (lat/long) to EPSG:31468
transformer = Transformer.from_crs("EPSG:4326", "EPSG:31468", always_xy=True)

# Define your 5 places here (name, latitude, longitude)
places = [
    ("Munich", 48.1375, 11.575),  # Example: Munich
    ("Ingolstadt", 48.763056, 11.425),  # Example: Berlin
    ("Augsburg", 48.368889, 10.897778),   # Example: Frankfurt
    ("Rosenheim", 47.85, 12.133333),  # Example: Leipzig
    ("Landshut", 48.539722, 12.150833),   # Example: Karlsruhe
]

print("=" * 80)
print("Coordinate Conversion: Lat/Long (EPSG:4326) → EPSG:31468")
print("=" * 80)
print()

# Convert each place
for name, lat, lon in places:
    # Transform coordinates (note: transformer expects lon, lat order)
    x, y = transformer.transform(lon, lat)
    
    print(f"{name}:")
    print(f"  Input  (Lat, Long): {lat:.6f}, {lon:.6f}")
    print(f"  Output (X, Y):      {x:.2f}, {y:.2f}")
    print()

print("=" * 80)
print("Note: Replace the example coordinates with your actual coordinates")
print("=" * 80)

