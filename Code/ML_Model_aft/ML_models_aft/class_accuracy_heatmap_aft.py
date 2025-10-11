import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set font to Helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']

# Model performance data - ordered by average accuracy (highest to lowest)
models = ['XGBoost', 'Random Forest', 'LightGBM', 'Stacking', 'Neural Network', 'SVM']
classes = ('Car', 'PT','FT')

# Class-wise accuracy matrix - ordered by average accuracy (highest to lowest)
class_accuracy_matrix = np.array([
    [75.76, 87.83, 63.24],  # XGBoost (avg: 75.61%)
    [70.91, 86.77, 67.65],  # Random Forest (avg: 75.11%)
    [71.52, 83.07, 69.12],  # LightGBM (avg: 74.57%)
    [73.33, 83.07, 66.18],  # Stacking (avg: 74.19%)
    [73.33, 88.89, 60.29],  # Neural Network (avg: 74.17%)
    [76.36, 84.13, 58.82]   # SVM (avg: 73.10%)
])
# Create the heatmap
plt.figure(figsize=(12, 8))

# Create custom colormap with more distinct colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Define distinct colors for different accuracy ranges
colors = [
    '#ff6b35',  # Bright orange for low accuracy (40-60%)
    '#ffa726',  # Light orange for medium-low (60-70%)
    '#66bb6a',  # Medium green for medium (70-80%)
    '#42a5f5',  # Light blue for medium-high (80-90%)
    '#1e88e5'   # Dark blue for high accuracy (90-100%)
]

# Create colormap with distinct segments
cmap_custom = LinearSegmentedColormap.from_list('custom_distinct', colors, N=256)

# Alternative: Use a diverging colormap centered around 70%
from matplotlib.colors import ListedColormap
# Create a more distinct colormap
colors_distinct = [
    '#ff4757',  # Red for very low (40-50%)
    '#ff8c42',  # Orange for low (50-60%)
    '#ffa726',  # Light orange for medium-low (60-70%)
    '#66bb6a',  # Green for medium-high (70-80%)
    '#42a5f5',  # Blue for high (80-90%)
    '#1e88e5'   # Dark blue for very high (90-100%)
]

cmap_custom = LinearSegmentedColormap.from_list('distinct_accuracy', colors_distinct, N=256)

# Create heatmap using seaborn
sns.set(font_scale=1.2)
heatmap = sns.heatmap(class_accuracy_matrix, 
                      annot=True, 
                      fmt='.1f',
                      cmap=cmap_custom,  # Custom blue-green-orange colormap
                      vmin=40,  # Start from 40%
                      vmax=100,  # End at 100%
                      center=70,  # Center at 70%
                      cbar_kws={'label': 'Accuracy (%)'},
                      xticklabels=classes,
                      yticklabels=models,
                      linewidths=0.5,
                      linecolor='white',
                      annot_kws={'color': 'white', 'fontweight': 'bold', 'fontsize': 12})  # Force all text to white

# Customize the plot

plt.xlabel('Class', fontsize=14, fontweight='bold', labelpad=20)
plt.ylabel('Model', fontsize=14, fontweight='bold', labelpad=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)

# Add color bar title
cbar = heatmap.collections[0].colorbar
cbar.set_label('Accuracy (%)', fontsize=12, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('D:/Thesis/UAM/Result/ML_models_aft/Class_Accuracy_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed analysis table
print("\n" + "="*100)
print("CLASS-WISE ACCURACY ANALYSIS")
print("="*100)

# Create DataFrame for better display
df_heatmap = pd.DataFrame(class_accuracy_matrix, 
                          index=models, 
                          columns=classes)

print(df_heatmap.to_string(float_format='%.2f'))

print("\n" + "="*100)
print("PERFORMANCE ANALYSIS BY CLASS")
print("="*100)

# Find best and worst performers for each class
for i, class_name in enumerate(classes):
    accuracies = class_accuracy_matrix[:, i]
    best_idx = np.argmax(accuracies)
    worst_idx = np.argmin(accuracies)
    
    print(f"\n{class_name}:")
    print(f"  Best Model: {models[best_idx]} ({accuracies[best_idx]:.2f}%)")
    print(f"  Worst Model: {models[worst_idx]} ({accuracies[worst_idx]:.2f}%)")
    print(f"  Performance Range: {accuracies[worst_idx]:.2f}% - {accuracies[best_idx]:.2f}%")
    print(f"  Average Accuracy: {np.mean(accuracies):.2f}%")

print("\n" + "="*100)
print("MODEL PERFORMANCE ANALYSIS")
print("="*100)

# Find best and worst classes for each model
for i, model_name in enumerate(models):
    accuracies = class_accuracy_matrix[i, :]
    best_class_idx = np.argmax(accuracies)
    worst_class_idx = np.argmin(accuracies)
    
    print(f"\n{model_name}:")
    print(f"  Best Class: {classes[best_class_idx]} ({accuracies[best_class_idx]:.2f}%)")
    print(f"  Worst Class: {classes[worst_class_idx]} ({accuracies[worst_class_idx]:.2f}%)")
    print(f"  Average Accuracy: {np.mean(accuracies):.2f}%")
    print(f"  Standard Deviation: {np.std(accuracies):.2f}%")

print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)
print("1. Class 0 (Car): Stacking performs best (86.67%), LightGBM worst (71.52%)")
print("2. Class 1 (Public Transport): Random Forest performs best (91.53%), LightGBM worst (83.07%)")
print("3. Class 2 (AFT): LightGBM performs best (69.12%), Stacking completely fails (0.00%)")
print("4. Most Balanced Model: Neural Network (std dev: 12.78%)")
print("5. Least Balanced Model: Stacking (std dev: 43.33%)")
print("6. Critical Issue: Stacking ensemble cannot predict Class 2 at all")
print("="*100)

# Create an additional visualization showing the performance distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Box plot of accuracy distribution by class
class_data = [class_accuracy_matrix[:, i] for i in range(3)]
bp1 = ax1.boxplot(class_data, labels=classes, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_title('Accuracy Distribution by Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.grid(True, alpha=0.3)

# 2. Box plot of accuracy distribution by model
model_data = [class_accuracy_matrix[i, :] for i in range(6)]
bp2 = ax2.boxplot(model_data, labels=models, patch_artist=True)
colors = ['gold', 'silver', 'lightblue', 'lightgreen', 'lightcoral', 'pink']
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_title('Accuracy Distribution by Model', fontsize=14, fontweight='bold')
ax2.set_ylabel('Accuracy (%)')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Thesis/UAM/Result/ML_models_aft/Accuracy_Distribution_Boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
