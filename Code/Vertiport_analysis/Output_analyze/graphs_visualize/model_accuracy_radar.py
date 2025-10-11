import matplotlib.pyplot as plt
import numpy as np

# Model accuracy data
models = ['LGBM', 'Stacking', 'RF', 'NN', 'SVM', 'XGB', 'MNL']
accuracies = [0.763, 0.7654, 0.7749, 0.782, 0.7701, 0.7915, 0.6872]

# Number of variables
num_vars = len(models)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the circle
accuracies += accuracies[:1]
angles += angles[:1]

# Create figure
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, accuracies, 'o-', linewidth=3, color='#4472C4', markersize=10)
ax.fill(angles, accuracies, alpha=0.3, color='#4472C4')

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(models, fontsize=14, fontweight='bold')

# Set y-axis range and labels
ax.set_ylim(0.66, 0.82)
ax.set_yticks([0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81])
ax.set_yticklabels(['0.67', '0.69', '0.71', '0.73', '0.75', '0.77', '0.79', '0.81'], 
                    fontsize=12, fontweight='bold')

# Add value labels at each point
for angle, accuracy, model in zip(angles[:-1], accuracies[:-1], models):
    ax.text(angle, accuracy + 0.01, f'{accuracy:.4f}', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='black', alpha=0.9, linewidth=1))

# Add grid
ax.grid(True, linestyle='--', alpha=0.7, linewidth=1, color='gray')

plt.tight_layout()

# Save figure
plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/graphs_visualize/model_accuracy_radar.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("=" * 80)
print("MODEL ACCURACY COMPARISON - RADAR CHART")
print("=" * 80)
print("\nAccuracy Summary:")
for model, acc in zip(models, accuracies[:-1]):
    print(f"  {model:20s}: {acc:.4f} ({acc*100:.2f}%)")

# Find best and worst
best_idx = accuracies[:-1].index(max(accuracies[:-1]))
worst_idx = accuracies[:-1].index(min(accuracies[:-1]))

print(f"\n✓ Best Model: {models[best_idx]} with {accuracies[best_idx]:.4f} accuracy")
print(f"✗ Lowest Model: {models[worst_idx]} with {accuracies[worst_idx]:.4f} accuracy")
print(f"  Difference: {accuracies[best_idx] - accuracies[worst_idx]:.4f} ({(accuracies[best_idx] - accuracies[worst_idx])*100:.2f}%)")

print("\n" + "=" * 80)
print("Radar chart saved: model_accuracy_radar.png")
print("=" * 80)

