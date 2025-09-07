import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Model performance data
models = ['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network', 'SVM', 'Stacking']
classes = ['Class 0 (Car)', 'Class 1 (Public Transport)', 'Class 2 (AFT)']

# Class-wise accuracy matrix
class_accuracy_matrix = np.array([
    [78.18, 91.53, 51.47],  # Random Forest
    [79.39, 89.42, 45.59],  # XGBoost
    [71.52, 83.07, 69.12],  # LightGBM
    [73.33, 88.89, 60.29],  # Neural Network
    [76.36, 84.13, 58.82],  # SVM
    [86.67, 86.24, 0.00]    # Stacking
])

# Create the heatmap
plt.figure(figsize=(12, 8))

# Create heatmap using seaborn
sns.set(font_scale=1.2)
heatmap = sns.heatmap(class_accuracy_matrix, 
                      annot=True, 
                      fmt='.1f',
                      cmap='RdYlGn',  # Red-Yellow-Green colormap
                      cbar_kws={'label': 'Accuracy (%)'},
                      xticklabels=classes,
                      yticklabels=models,
                      linewidths=0.5,
                      linecolor='white')

# Customize the plot
plt.title('Class-wise Accuracy Heatmap for All Models_oldSurveydata', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Transportation Classes', fontsize=14, fontweight='bold')
plt.ylabel('Machine Learning Models_oldSurveydata', fontsize=14, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add color bar title
cbar = heatmap.collections[0].colorbar
cbar.set_label('Accuracy (%)', fontsize=12, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('Class_Accuracy_Heatmap.png', dpi=300, bbox_inches='tight')
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
ax2.tick_params(axis='x', rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Accuracy_Distribution_Boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
