import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis('off')

# Define colors
colors = {
    'start_end': '#E8F4FD',
    'data': '#FFF2CC',
    'model': '#D5E8D4',
    'evaluation': '#F8CECC',
    'output': '#E1D5E7',
    'decision': '#FFE6CC'
}

# Function to create rounded rectangle boxes
def create_box(x, y, width, height, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, 
            weight='bold', wrap=True)

# Function to create arrows
def create_arrow(start_x, start_y, end_x, end_y, label=""):
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), 
                           "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc="black", linewidth=1.5)
    ax.add_patch(arrow)
    if label:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom', 
                fontsize=8, weight='bold')

# Title
ax.text(5, 24.5, 'Random Forest Model Training Flowchart', 
        ha='center', va='center', fontsize=16, weight='bold')

# 1. START
create_box(4, 23, 2, 0.8, "START", colors['start_end'], 12)

# 2. Data Loading and Setup
create_box(0.5, 21.5, 9, 1, "Data Loading & Setup\n• Load UAM survey data\n• Set random seeds (42)\n• Setup logging\n• Define features (X) and target (y)\n• Extract unique classes", colors['data'])

# 3. Pipeline Creation
create_box(0.5, 20, 9, 1, "Pipeline Creation\n• SimpleImputer (fill_value=0)\n• RandomForestClassifier\n• Hyperparameter grid definition", colors['model'])

# 4. Data Splitting
create_box(0.5, 18.5, 9, 1, "Data Splitting\n• Train+Val (80%) / Test (20%)\n• Stratified split\n• Setup 10-fold StratifiedKFold CV", colors['data'])

# 5. Initialize Metrics Storage
create_box(0.5, 17, 9, 1, "Initialize Metrics Storage\n• accuracies, precisions, recalls\n• f1s, roc_aucs, confusion_matrices\n• probabilities, feature_importances\n• best_params, class_accuracies", colors['evaluation'])

# 6. Cross-Validation Loop Start
create_box(3, 15.5, 4, 0.8, "10-Fold Cross-Validation Loop", colors['decision'], 10)

# 7. Fold Processing (Left side)
create_box(0.5, 14, 4, 1, "For each fold:\n• Split train/val data\n• GridSearchCV (5-fold)\n• Hyperparameter tuning\n• Best model selection", colors['model'])

# 8. Model Training (Left side)
create_box(0.5, 12.5, 4, 1, "Model Training\n• Fit best model on train\n• Calculate training accuracy\n• Store feature importances", colors['model'])

# 9. Validation (Left side)
create_box(0.5, 11, 4, 1, "Validation\n• Predict on validation set\n• Calculate metrics (acc, prec, rec, f1)\n• Calculate ROC AUC\n• Per-class accuracy", colors['evaluation'])

# 10. Metrics Storage (Left side)
create_box(0.5, 9.5, 4, 1, "Store Results\n• Append all metrics\n• Store probabilities\n• Store confusion matrix\n• Store best parameters", colors['evaluation'])

# 11. Fold Processing (Right side)
create_box(5.5, 14, 4, 1, "GridSearchCV Process\n• n_estimators: [70,80,90]\n• max_depth: [8,10,12]\n• min_samples_split: [5,8,10]\n• min_samples_leaf: [2,3,4]", colors['model'])

# 12. Hyperparameter Tuning (Right side)
create_box(5.5, 12.5, 4, 1, "Hyperparameter Tuning\n• 5-fold CV for tuning\n• Scoring: accuracy\n• Parallel processing (n_jobs=-1)\n• Best estimator selection", colors['model'])

# 13. Model Evaluation (Right side)
create_box(5.5, 11, 4, 1, "Model Evaluation\n• Predict probabilities\n• Calculate all metrics\n• Handle ROC AUC errors\n• Log results", colors['evaluation'])

# 14. Loop End Check
create_box(3, 8.5, 4, 0.8, "All 10 folds completed?", colors['decision'], 10)

# 15. Post-CV Processing
create_box(0.5, 7, 9, 1, "Post-CV Processing\n• Concatenate all probabilities\n• Save to CSV\n• Parameter stability analysis\n• Find most common best parameters", colors['evaluation'])

# 16. Final Model Training
create_box(0.5, 5.5, 9, 1, "Final Model Training\n• Train on full train+val data\n• Use most common best parameters\n• Calculate training accuracy", colors['model'])

# 17. Test Set Evaluation
create_box(0.5, 4, 9, 1, "Test Set Evaluation\n• Predict on test set\n• Calculate all metrics\n• Per-class accuracy\n• Save test probabilities", colors['evaluation'])

# 18. Feature Importance Analysis
create_box(0.5, 2.5, 9, 1, "Feature Importance Analysis\n• Calculate mean feature importance\n• Sort by importance\n• Save to CSV\n• Top 10 features", colors['evaluation'])

# 19. Results Saving
create_box(0.5, 1, 9, 1, "Results Saving\n• Save comprehensive results to txt\n• Save confusion matrix to CSV\n• Log completion message", colors['output'])

# 20. END
create_box(4, 0.2, 2, 0.8, "END", colors['start_end'], 12)

# Create arrows
# Main flow
create_arrow(5, 23, 5, 22.3)  # Start to Data Loading
create_arrow(5, 21.5, 5, 20.8)  # Data Loading to Pipeline
create_arrow(5, 20, 5, 19.3)  # Pipeline to Data Splitting
create_arrow(5, 18.5, 5, 17.8)  # Data Splitting to Initialize
create_arrow(5, 17, 5, 16.3)  # Initialize to CV Loop
create_arrow(5, 15.5, 5, 14.8)  # CV Loop to Fold Processing
create_arrow(5, 14, 5, 13.3)  # Fold Processing to Model Training
create_arrow(5, 12.5, 5, 11.8)  # Model Training to Validation
create_arrow(5, 11, 5, 10.3)  # Validation to Store Results
create_arrow(5, 9.5, 5, 9.3)  # Store Results to Loop Check
create_arrow(5, 8.5, 5, 7.8)  # Loop Check to Post-CV
create_arrow(5, 7, 5, 6.3)  # Post-CV to Final Model
create_arrow(5, 5.5, 5, 4.8)  # Final Model to Test Evaluation
create_arrow(5, 4, 5, 3.3)  # Test Evaluation to Feature Importance
create_arrow(5, 2.5, 5, 1.8)  # Feature Importance to Results
create_arrow(5, 1, 5, 1)  # Results to End

# Loop arrows
create_arrow(2.5, 8.5, 2.5, 14.8, "No")  # Loop back
create_arrow(7.5, 8.5, 7.5, 14.8, "No")  # Loop back

# Side connections
create_arrow(4.5, 14, 5.5, 14, "")  # Connect fold processing
create_arrow(4.5, 12.5, 5.5, 12.5, "")  # Connect model training
create_arrow(4.5, 11, 5.5, 11, "")  # Connect validation

# Add legend
legend_elements = [
    patches.Patch(color=colors['start_end'], label='Start/End'),
    patches.Patch(color=colors['data'], label='Data Processing'),
    patches.Patch(color=colors['model'], label='Model Operations'),
    patches.Patch(color=colors['evaluation'], label='Evaluation'),
    patches.Patch(color=colors['output'], label='Output'),
    patches.Patch(color=colors['decision'], label='Decision Points')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Add some annotations for key features
ax.annotate('Key Features:', xy=(0.02, 0.95), xycoords='axes fraction', 
            fontsize=10, weight='bold')
ax.annotate('• 10-fold stratified cross-validation\n• GridSearchCV for hyperparameter tuning\n• Comprehensive metrics calculation\n• Feature importance analysis\n• Parameter stability analysis\n• Overfitting detection', 
            xy=(0.02, 0.85), xycoords='axes fraction', fontsize=9)

plt.tight_layout()
plt.savefig('random_forest_flowchart.png', dpi=300, bbox_inches='tight')
plt.show()

print("Random Forest flowchart has been generated and saved as 'random_forest_flowchart.png'")
