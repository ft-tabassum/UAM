import matplotlib.pyplot as plt
import numpy as np

# Data
studies = ['OBUAM', 'KM++', 'Weighted KM++']
demand_covered = [27.20, 52.00, 58.80]

# Create figure with larger size
plt.figure(figsize=(12, 8))

# Create bar chart with outline only (no fill)
x_pos = np.arange(len(studies))
colors = ['#4472C4', '#ED7D31', '#70AD47']  # Different outline colors
bars = plt.bar(x_pos, demand_covered, 
               facecolor='none',  # No fill
               edgecolor=colors,  # Colored outlines
               linewidth=3,  # Thick outlines
               width=0.6)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, demand_covered)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.2f}%', ha='center', va='bottom', 
             fontsize=16, fontweight='bold')

# Customize axes and labels
plt.ylabel('Demand Coverage Ratio(%)', fontsize=18, fontweight='bold')
plt.xticks(x_pos, studies, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16)

# Add grid for better readability
plt.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Set y-axis limit
plt.ylim(0, max(demand_covered) + 10)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/demand_point_coverage_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved successfully!")
print("\nSummary:")
for study, coverage in zip(studies, demand_covered):
    print(f"{study:20s}: {coverage:6.2f}%")

