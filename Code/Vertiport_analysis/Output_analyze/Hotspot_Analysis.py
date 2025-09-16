import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

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

# Create separate figures for each visualization approach

# 1. HEXBIN DENSITY PLOT (Most Recommended)
plt.figure(figsize=(12, 10))
hexbin = plt.hexbin(od_points[:, 0], od_points[:, 1], gridsize=50, cmap='Blues', alpha=0.8)
plt.scatter(od_points[:, 0], od_points[:, 1], color='red', s=2, alpha=0.3, label='O-D trips')
plt.scatter(vertiport_coords_KMPlus['x'], vertiport_coords_KMPlus['y'], 
           color='green', marker='X', s=80, label='KMPlus Vertiports', edgecolors='black', linewidth=1)
plt.scatter(vertiport_coords_OBUAM['x'], vertiport_coords_OBUAM['y'], 
           color='blue', marker='X', s=80, label='OBUAM Vertiports', edgecolors='black', linewidth=1)
plt.scatter(demand_vertiport_coords['x'], demand_vertiport_coords['y'], 
           color='orange', marker='X', s=80, label='Demand Vertiports', edgecolors='black', linewidth=1)
plt.title('Hexbin Density Plot - Trip Hotspots', fontsize=16)
plt.xlabel('X Coordinate (meters)', fontsize=12)
plt.ylabel('Y Coordinate (meters)', fontsize=12)
plt.legend()
plt.axis('equal')
plt.colorbar(hexbin, label='Trip Count per Hex')
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/hotspot_hexbin.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. GRID-BASED HEATMAP
plt.figure(figsize=(12, 10))
# Create grid
x_min, x_max = od_points[:, 0].min(), od_points[:, 0].max()
y_min, y_max = od_points[:, 1].min(), od_points[:, 1].max()
grid_size = 30
x_bins = np.linspace(x_min, x_max, grid_size)
y_bins = np.linspace(y_min, y_max, grid_size)

# Count points in each grid cell
H, xedges, yedges = np.histogram2d(od_points[:, 0], od_points[:, 1], bins=[x_bins, y_bins])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

im = plt.pcolormesh(X, Y, H.T, cmap='Reds', alpha=0.8)
plt.scatter(vertiport_coords_KMPlus['x'], vertiport_coords_KMPlus['y'], 
           color='green', marker='X', s=80, label='KMPlus Vertiports', edgecolors='black', linewidth=1)
plt.scatter(vertiport_coords_OBUAM['x'], vertiport_coords_OBUAM['y'], 
           color='blue', marker='X', s=80, label='OBUAM Vertiports', edgecolors='black', linewidth=1)
plt.scatter(demand_vertiport_coords['x'], demand_vertiport_coords['y'], 
           color='orange', marker='X', s=80, label='Demand Vertiports', edgecolors='black', linewidth=1)
plt.title('Grid-based Heatmap - Trip Distribution', fontsize=16)
plt.xlabel('X Coordinate (meters)', fontsize=12)
plt.ylabel('Y Coordinate (meters)', fontsize=12)
plt.legend()
plt.axis('equal')
plt.colorbar(im, label='Trip Count per Grid Cell')
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/hotspot_grid_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. IMPROVED KDE WITH BETTER SETTINGS
plt.figure(figsize=(12, 10))
from sklearn.neighbors import KernelDensity

# Calculate appropriate bandwidth
x_range = od_points[:, 0].max() - od_points[:, 0].min()
y_range = od_points[:, 1].max() - od_points[:, 1].min()
bandwidth = min(x_range, y_range) * 0.03  # Slightly larger bandwidth

# Apply KDE
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
kde.fit(od_points)

# Create grid
margin = bandwidth * 2
x_min_kde, x_max_kde = od_points[:, 0].min() - margin, od_points[:, 0].max() + margin
y_min_kde, y_max_kde = od_points[:, 1].min() - margin, od_points[:, 1].max() + margin
x_grid, y_grid = np.meshgrid(np.linspace(x_min_kde, x_max_kde, 150),
                             np.linspace(y_min_kde, y_max_kde, 150))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Evaluate KDE
z = np.exp(kde.score_samples(grid_points))
z = z.reshape(x_grid.shape)

contour = plt.contourf(x_grid, y_grid, z, levels=25, cmap='viridis', alpha=0.7)
plt.scatter(od_points[:, 0], od_points[:, 1], color='red', s=3, alpha=0.4, label='O-D trips')
plt.scatter(vertiport_coords_KMPlus['x'], vertiport_coords_KMPlus['y'], 
           color='green', marker='X', s=80, label='KMPlus Vertiports', edgecolors='black', linewidth=1)
plt.scatter(vertiport_coords_OBUAM['x'], vertiport_coords_OBUAM['y'], 
           color='blue', marker='X', s=80, label='OBUAM Vertiports', edgecolors='black', linewidth=1)
plt.scatter(demand_vertiport_coords['x'], demand_vertiport_coords['y'], 
           color='orange', marker='X', s=80, label='Demand Vertiports', edgecolors='black', linewidth=1)
plt.title('Improved KDE - Smooth Density Estimation', fontsize=16)
plt.xlabel('X Coordinate (meters)', fontsize=12)
plt.ylabel('Y Coordinate (meters)', fontsize=12)
plt.legend()
plt.axis('equal')
plt.colorbar(contour, label='Density')
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/hotspot_kde_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. SCATTER PLOT WITH TRANSPARENCY LAYERS
plt.figure(figsize=(12, 10))
# Plot multiple layers with different transparency
plt.scatter(od_points[:, 0], od_points[:, 1], color='red', s=8, alpha=0.1, label='O-D trips (low alpha)')
plt.scatter(od_points[:, 0], od_points[:, 1], color='red', s=4, alpha=0.3, label='O-D trips (medium alpha)')
plt.scatter(od_points[:, 0], od_points[:, 1], color='red', s=2, alpha=0.6, label='O-D trips (high alpha)')

plt.scatter(vertiport_coords_KMPlus['x'], vertiport_coords_KMPlus['y'], 
           color='green', marker='X', s=100, label='KMPlus Vertiports', edgecolors='black', linewidth=1)
plt.scatter(vertiport_coords_OBUAM['x'], vertiport_coords_OBUAM['y'], 
           color='blue', marker='X', s=100, label='OBUAM Vertiports', edgecolors='black', linewidth=1)
plt.scatter(demand_vertiport_coords['x'], demand_vertiport_coords['y'], 
           color='orange', marker='X', s=100, label='Demand Vertiports', edgecolors='black', linewidth=1)
plt.title('Multi-layer Scatter Plot - Density through Transparency', fontsize=16)
plt.xlabel('X Coordinate (meters)', fontsize=12)
plt.ylabel('Y Coordinate (meters)', fontsize=12)
plt.legend()
plt.axis('equal')
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/hotspot_multilayer_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization Summary:")
print(f"- Hexbin plot: Best for showing density patterns")
print(f"- Grid heatmap: Most intuitive for understanding trip distribution")
print(f"- Improved KDE: Smoother density estimation")
print(f"- Multi-layer scatter: Shows data density through transparency")
print(f"\nAll visualizations saved as separate PNG files:")
print(f"1. hotspot_hexbin.png")
print(f"2. hotspot_grid_heatmap.png")
print(f"3. hotspot_kde_improved.png")
print(f"4. hotspot_multilayer_scatter.png")
