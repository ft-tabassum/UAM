import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Load your datasets
od_trip_data = pd.read_csv('D:/Files_D/Study/Thesis/new_data/Analysis_syntheticpop/microdata_trips.csv')
demand_vertiport_coords = pd.read_csv('D:/Files_D/Study/Thesis/new_data/Analysis_syntheticpop/demand_vertiport_coords.csv')
vertiport_coords_KMPlus = pd.read_csv('D:/Files_D/Study/Thesis/new_data/Analysis_syntheticpop/vertiport_coords_KMPlus.csv')
vertiport_coords_OBUAM = pd.read_csv('D:/Files_D/Study/Thesis/new_data/Analysis_syntheticpop/vertiport_coords_OBUAM.csv')

# Combine origin and destination coordinates from OD trip data
od_points = np.vstack([
    od_trip_data[['originX', 'originY']].values,
    od_trip_data[['destinationX', 'destinationY']].values
])

print(f"Total OD points to plot: {len(od_points)}")
print(f"OD points range - X: {od_points[:, 0].min():.2f} to {od_points[:, 0].max():.2f}")
print(f"OD points range - Y: {od_points[:, 1].min():.2f} to {od_points[:, 1].max():.2f}")

# Calculate appropriate bandwidth based on data scale
x_range = od_points[:, 0].max() - od_points[:, 0].min()
y_range = od_points[:, 1].max() - od_points[:, 1].min()
bandwidth = min(x_range, y_range) * 0.02  # 2% of the smaller range
print(f"Data range - X: {x_range:.0f}m, Y: {y_range:.0f}m")
print(f"Using bandwidth: {bandwidth:.0f}m")

# Apply Kernel Density Estimation (KDE)
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
kde.fit(od_points)

# Create a grid of points to evaluate the KDE on
margin = bandwidth * 2  # Add margin based on bandwidth
x_min, x_max = od_points[:, 0].min() - margin, od_points[:, 0].max() + margin
y_min, y_max = od_points[:, 1].min() - margin, od_points[:, 1].max() + margin

# Create grid for visualization with higher resolution
grid_size = 200  # Increased from 100 for better visualization
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                             np.linspace(y_min, y_max, grid_size))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Evaluate KDE on the grid
z = np.exp(kde.score_samples(grid_points))
z = z.reshape(x_grid.shape)

# Plot the KDE (Hotspot) analysis
plt.figure(figsize=(12, 10))
plt.contourf(x_grid, y_grid, z, levels=30, cmap='Blues_r', alpha=0.8)
plt.colorbar(label='Density')
plt.scatter(od_points[:, 0], od_points[:, 1], color='red', s=15, alpha=0.8, label='O-D trips')
plt.title("O-D Trip Density Hotspots", fontsize=16)
plt.xlabel("X Coordinate (meters)", fontsize=12)
plt.ylabel("Y Coordinate (meters)", fontsize=12)
plt.axis('equal')  # Equal aspect ratio for proper geographic visualization

# Overlay vertiport locations for each dataset with different colors
plt.scatter(vertiport_coords_KMPlus['x'], vertiport_coords_KMPlus['y'], color='green', marker='X', s=100, label='KMPlus Vertiports', edgecolors='black', linewidth=1)
plt.scatter(vertiport_coords_OBUAM['x'], vertiport_coords_OBUAM['y'], color='blue', marker='X', s=100, label='OBUAM Vertiports', edgecolors='black', linewidth=1)
plt.scatter(demand_vertiport_coords['x'], demand_vertiport_coords['y'], color='orange', marker='X', s=100, label='Demand Vertiports', edgecolors='black', linewidth=1)

plt.legend()
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/hotspot_analysis.png', dpi=300)  # Save as PNG with high resolution (300 DPI)
plt.show()
