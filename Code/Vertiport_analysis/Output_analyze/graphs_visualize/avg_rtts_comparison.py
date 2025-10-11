import matplotlib.pyplot as plt
import numpy as np

# Data
scenarios = ['KM++', 'Weighted KM++']
avg_rtts = [10.05, 11.08]

# Create figure with larger size
plt.figure(figsize=(10, 8))

# Create bar chart
x_pos = np.arange(len(scenarios))
bars = plt.bar(x_pos, avg_rtts, color=['#ED7D31', '#70AD47'], 
               alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, avg_rtts)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{value:.2f}%', ha='center', va='bottom', 
             fontsize=16, fontweight='bold')

# Customize axes and labels
plt.ylabel('Average RTTs (%)', fontsize=18, fontweight='bold')
plt.xlabel('Scenario', fontsize=18, fontweight='bold')
plt.xticks(x_pos, scenarios, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16)

# Add grid for better readability
plt.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Set y-axis limit
plt.ylim(0, max(avg_rtts) + 2)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/graphs_visualize/avg_rtts_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("=" * 80)
print("AVERAGE RTTs COMPARISON")
print("=" * 80)
print("\nSummary:")
for scenario, rtts in zip(scenarios, avg_rtts):
    print(f"{scenario:20s}: {rtts:6.2f}%")

# Calculate difference
difference = avg_rtts[1] - avg_rtts[0]
percentage_increase = (difference / avg_rtts[0]) * 100

print(f"\n✓ Weighted KM++ has {difference:.2f}% higher RTTs than KM++")
print(f"  ({percentage_increase:.2f}% increase)")

print("\n" + "=" * 80)
print("Visualization saved: avg_rtts_comparison.png")
print("=" * 80)

