import matplotlib.pyplot as plt
import numpy as np

# Data
scenarios = ['KM++', 'Weighted KM++']
avg_rtts = [10.05, 11.08]

# Create output directory path
output_dir = '/Result/Vertiport_analysis/Output_analyze/graphs_visualize/'

# =============================================================================
# OPTION 1: HORIZONTAL BAR CHART (easier to read labels)
# =============================================================================
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(scenarios))
bars = plt.barh(y_pos, avg_rtts, color=['#ED7D31', '#70AD47'], 
                alpha=0.85, edgecolor='black', linewidth=1.5, height=0.5)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, avg_rtts)):
    plt.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2, 
             f'{value:.2f}%', ha='left', va='center', 
             fontsize=16, fontweight='bold')

plt.xlabel('Average rtts (%)', fontsize=18, fontweight='bold')
plt.yticks(y_pos, scenarios, fontsize=16, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
plt.xlim(0, max(avg_rtts) + 2)
plt.tight_layout()
plt.savefig(output_dir + 'avg_rtts_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 1 saved: avg_rtts_horizontal.png")

# =============================================================================
# OPTION 2: LOLLIPOP CHART (modern, clean look) - ADJUSTED SCALE & SPACING
# =============================================================================
plt.figure(figsize=(10, 5))  # Reduced height to bring bars closer
y_pos = np.arange(len(scenarios))
colors = ['#ED7D31', '#70AD47']

# Create lollipop chart with adjusted x-axis
for i, (scenario, value, color) in enumerate(zip(scenarios, avg_rtts, colors)):
    plt.hlines(y=i, xmin=9.5, xmax=value, color=color, linewidth=4, alpha=0.8)
    plt.plot(value, i, 'o', markersize=20, color=color, alpha=0.9, 
             markeredgecolor='black', markeredgewidth=2)
    plt.text(value + 0.15, i, f'{value:.2f}%', va='center', 
             fontsize=16, fontweight='bold')

plt.yticks(y_pos, scenarios, fontsize=16, fontweight='bold')
plt.xlabel('Average rtts (%)', fontsize=18, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
plt.xlim(9.5, max(avg_rtts) + 0.8)  # Zoomed in to show difference better
plt.ylim(-0.5, 1.5)  # Tighter vertical spacing
plt.tight_layout()
plt.savefig(output_dir + 'avg_rtts_lollipop.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 2 saved: avg_rtts_lollipop.png (adjusted scale & spacing)")

# =============================================================================
# OPTION 3: LINE CHART WITH MARKERS (showing improvement)
# =============================================================================
plt.figure(figsize=(10, 8))
x_pos = np.arange(len(scenarios))

plt.plot(x_pos, avg_rtts, 'o-', linewidth=4, markersize=18, 
         color='#4472C4', markeredgecolor='black', markeredgewidth=2)

# Add value labels
for i, value in enumerate(avg_rtts):
    plt.text(i, value + 0.3, f'{value:.2f}%', ha='center', va='bottom', 
             fontsize=16, fontweight='bold')

# Add arrow showing improvement
plt.annotate('', xy=(1, avg_rtts[1]), xytext=(0, avg_rtts[0]),
            arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.6))
plt.text(0.5, (avg_rtts[0] + avg_rtts[1])/2, 
         f'+{avg_rtts[1] - avg_rtts[0]:.2f}%\nimprovement', 
         ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.ylabel('Average rtts (%)', fontsize=18, fontweight='bold')
plt.xticks(x_pos, scenarios, fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylim(9, max(avg_rtts) + 1.5)
plt.tight_layout()
plt.savefig(output_dir + 'avg_rtts_line.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 3 saved: avg_rtts_line.png")

# =============================================================================
# OPTION 4: DIFFERENCE BAR (showing percentage increase)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Absolute values
x_pos = np.arange(len(scenarios))
bars = ax1.bar(x_pos, avg_rtts, color=['#ED7D31', '#70AD47'], 
               alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)
for i, (bar, value) in enumerate(zip(bars, avg_rtts)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{value:.2f}%', ha='center', va='bottom', 
             fontsize=14, fontweight='bold')
ax1.set_ylabel('Average rtts (%)', fontsize=16, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenarios, fontsize=14, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_title('Absolute Values', fontsize=14, fontweight='bold')

# Right: Improvement
difference = avg_rtts[1] - avg_rtts[0]
percentage_increase = (difference / avg_rtts[0]) * 100
bar = ax2.bar([0], [percentage_increase], color='#70AD47', 
              alpha=0.85, edgecolor='black', linewidth=1.5, width=0.4)
ax2.text(0, percentage_increase + 0.5, f'+{percentage_increase:.2f}%', 
         ha='center', va='bottom', fontsize=14, fontweight='bold')
ax2.set_ylabel('Improvement (%)', fontsize=16, fontweight='bold')
ax2.set_xticks([0])
ax2.set_xticklabels(['Weighted KM++\nvs KM++'], fontsize=12, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_title('Relative Improvement', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir + 'avg_rtts_comparison_split.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 4 saved: avg_rtts_comparison_split.png")

print("\n" + "=" * 80)
print("All 4 alternative visualizations created!")
print("=" * 80)
print("\nSummary:")
for scenario, rtts in zip(scenarios, avg_rtts):
    print(f"  {scenario:20s}: {rtts:6.2f}%")
print(f"\n  Improvement: +{difference:.2f}% ({percentage_increase:.2f}% increase)")

