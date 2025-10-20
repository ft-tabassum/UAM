import matplotlib.pyplot as plt
import numpy as np

# Mode share data
modes = ['Car', 'Public Transport', 'UAM']
percentages = [79.72, 17.00, 3.28]

# Create output directory path
output_dir = '/Result/Vertiport_analysis/Output_analyze/graphs_visualize/'

# Colors for each mode
colors = ['#ED7D31', '#4472C4', '#70AD47']

# =============================================================================
# OPTION 1: PIE CHART
# =============================================================================
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(percentages, labels=modes, autopct='%1.2f%%',
                                     colors=colors, startangle=90, 
                                     textprops={'fontsize': 16, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 2})

# Make percentage text larger
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(18)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig(output_dir + 'mode_share_pie.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 1 saved: mode_share_pie.png")

# =============================================================================
# OPTION 2: DONUT CHART (Modern look) - MUCH LARGER
# =============================================================================
plt.figure(figsize=(14, 14))
wedges, texts, autotexts = plt.pie(percentages, labels=modes, autopct='%1.2f%%',
                                     colors=colors, startangle=90,
                                     textprops={'fontsize': 20, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 2.5},
                                     pctdistance=0.85)

# Draw circle in center to create donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white', edgecolor='black', linewidth=3)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Make percentage text much larger and white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(24)
    autotext.set_fontweight('bold')

# Make label text larger
for text in texts:
    text.set_fontsize(22)
    text.set_fontweight('bold')

# Add center text much larger
plt.text(0, 0, 'Mode\nShare', ha='center', va='center', 
         fontsize=32, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir + 'mode_share_donut.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 2 saved: mode_share_donut.png (much larger)")

# =============================================================================
# OPTION 3: BAR CHART
# =============================================================================
plt.figure(figsize=(10, 8))
x_pos = np.arange(len(modes))
bars = plt.bar(x_pos, percentages, color=colors, alpha=0.85, 
               edgecolor='black', linewidth=2, width=0.6)

# Add value labels on top
for i, (bar, value) in enumerate(zip(bars, percentages)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
             f'{value:.2f}%', ha='center', va='bottom', 
             fontsize=16, fontweight='bold')

plt.ylabel('Mode Share (%)', fontsize=18, fontweight='bold')
plt.xlabel('Transport Mode', fontsize=18, fontweight='bold')
plt.xticks(x_pos, modes, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16)
plt.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
plt.ylim(0, max(percentages) + 10)
plt.tight_layout()
plt.savefig(output_dir + 'mode_share_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 3 saved: mode_share_bar.png")

# =============================================================================
# OPTION 4: HORIZONTAL BAR CHART
# =============================================================================
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(modes))
bars = plt.barh(y_pos, percentages, color=colors, alpha=0.85,
                edgecolor='black', linewidth=2, height=0.6)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, percentages)):
    plt.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, 
             f'{value:.2f}%', ha='left', va='center', 
             fontsize=16, fontweight='bold')

plt.xlabel('Mode Share (%)', fontsize=18, fontweight='bold')
plt.yticks(y_pos, modes, fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
plt.xlim(0, max(percentages) + 10)
plt.tight_layout()
plt.savefig(output_dir + 'mode_share_horizontal_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Option 4 saved: mode_share_horizontal_bar.png")

print("\n" + "=" * 80)
print("MODE SHARE ANALYSIS")
print("=" * 80)
print("\nMode Share:")
for mode, pct in zip(modes, percentages):
    print(f"  {mode:20s}: {pct:6.2f}%")
print(f"\n  Total: {sum(percentages):.2f}%")

print("\n" + "=" * 80)
print("All 4 visualizations created!")
print("=" * 80)

