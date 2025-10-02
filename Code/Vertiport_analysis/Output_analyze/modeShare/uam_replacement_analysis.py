import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set font to Arial for all plots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Load the data
data = pd.read_csv(
    'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

print("=" * 80)
print("UAM REPLACEMENT ANALYSIS VISUALIZATION")
print("=" * 80)

# Step 1: Filter for UAM trips only (where UAM is the chosen mode)
uam_trips = data[data['prob_mode_Autonomous Flying Taxi'] > data[['prob_mode_Car', 'prob_mode_Public Transport']].max(axis=1)]

print(f"Total UAM trips: {len(uam_trips):,}")

# Step 2: Determine what mode UAM is replacing
uam_trips['uam_replaces_pt'] = (
    (uam_trips['prob_mode_Autonomous Flying Taxi'] > uam_trips['prob_mode_Public Transport']) &
    (uam_trips['prob_mode_Public Transport'] > uam_trips['prob_mode_Car'])
)

uam_trips['uam_replaces_car'] = (
    (uam_trips['prob_mode_Autonomous Flying Taxi'] > uam_trips['prob_mode_Car']) &
    (uam_trips['prob_mode_Car'] > uam_trips['prob_mode_Public Transport'])
)

# Calculate replacement statistics
pt_replacement = uam_trips['uam_replaces_pt'].sum()
car_replacement = uam_trips['uam_replaces_car'].sum()
total_uam = len(uam_trips)

print(f"UAM trips replacing PT: {pt_replacement:,} ({pt_replacement/total_uam*100:.1f}%)")
print(f"UAM trips replacing Car: {car_replacement:,} ({car_replacement/total_uam*100:.1f}%)")

# Step 3: Create separate clear visualizations

# 1. Pie chart showing UAM replacement
plt.figure(figsize=(10, 8))
replacement_data = [pt_replacement, car_replacement]
replacement_labels = ['Replacing PT\n(16.5%)', 'Replacing Car\n(83.5%)']
colors = ['#ff7f0e', '#1f77b4']
explode = (0.05, 0)  # Slightly separate the PT slice

plt.pie(replacement_data, labels=replacement_labels, autopct='%1.1f%%', 
        colors=colors, startangle=90, explode=explode, shadow=True)
plt.title('UAM Trips: Which Mode Are They Replacing?', #\n\nTotal UAM Trips: 32,740
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
pie_path = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/uam_replacement_pie.png'
plt.savefig(pie_path, dpi=300, bbox_inches='tight')
print(f"Pie chart saved to: {pie_path}")
plt.show()

# 2. Bar chart showing absolute numbers
plt.figure(figsize=(10, 8))
bars = plt.bar(replacement_labels, replacement_data, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=2)
plt.title('UAM Replacement: Absolute Numbers\n\nTotal UAM Trips: 32,740', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of Trips', fontsize=14, fontweight='bold')
plt.xlabel('Replaced Mode', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, replacement_data)):
    plt.text(bar.get_x() + bar.get_width()/2, value + 800, 
             f'{value:,}\n({value/total_uam*100:.1f}%)', 
             ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
bar_path = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/uam_replacement_bars.png'
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
print(f"Bar chart saved to: {bar_path}")
plt.show()

# 3. Distance-based analysis
plt.figure(figsize=(12, 8))
distance_categories = ['Medium\n(20-50km)', 'Long\n(50-100km)', 'Very Long\n(>100km)']
pt_shares = [15.30, 71.41, 81.58]
car_shares = [100 - share for share in pt_shares]

x = np.arange(len(distance_categories))
width = 0.35

bars1 = plt.bar(x - width/2, pt_shares, width, label='Replacing PT', color='#ff7f0e', alpha=0.8, edgecolor='black')
bars2 = plt.bar(x + width/2, car_shares, width, label='Replacing Car', color='#1f77b4', alpha=0.8, edgecolor='black')

plt.title('UAM Replacement by Distance Category', #\n\nHow UAM Competition Changes with Distance
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Percentage (%)', fontsize=16, fontweight='bold')
plt.xlabel('Distance Category', fontsize=16, fontweight='bold')
plt.xticks(x, distance_categories, fontsize=16)
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 100)

# Add percentage labels on bars
for i, (pt, car) in enumerate(zip(pt_shares, car_shares)):
    plt.text(i - width/2, pt + 2, f'{pt:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
    plt.text(i + width/2, car + 2, f'{car:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.tight_layout()
distance_path = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/uam_replacement_by_distance.png'
plt.savefig(distance_path, dpi=300, bbox_inches='tight')
print(f"Distance analysis saved to: {distance_path}")
plt.show()

# 4. Summary infographic
plt.figure(figsize=(12, 8))
plt.axis('off')

summary_text = f"""UAM REPLACEMENT ANALYSIS SUMMARY

Total UAM Trips: {total_uam:,}

REPLACEMENT BREAKDOWN:
• Replacing Car: {car_replacement:,} ({car_replacement/total_uam*100:.1f}%)
• Replacing PT: {pt_replacement:,} ({pt_replacement/total_uam*100:.1f}%)

KEY INSIGHTS:
• Most UAM trips (83.5%) replace Car trips
• PT replacement is relatively low (16.5%)
• Long-distance trips show higher PT replacement
• UAM competes primarily with Car for shorter trips

PRICING IMPACT:
• €5.00 base fare + €2.00/km makes UAM expensive
• UAM competes with Car (also expensive) rather than PT (cheap)
• For long trips, time savings become more valuable than cost
"""

plt.text(0.5, 0.5, summary_text, transform=plt.gca().transAxes, fontsize=14, 
         verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8, edgecolor='navy'))

plt.title('UAM Replacement Analysis - Key Findings', fontsize=18, fontweight='bold', pad=30)
plt.tight_layout()
summary_path = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/modeShare/uam_replacement_summary.png'
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"Summary infographic saved to: {summary_path}")
plt.show()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
