import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_rf_overview_flowchart():
    """Create a high-level Random Forest flowchart without detailed CV breakdown"""
    
    # Create figure with better proportions
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Simple, clean colors
    start_color = '#90EE90'      # Light green for start/end
    process_color = '#87CEEB'    # Light blue for processes
    decision_color = '#FFB6C1'   # Light pink for decisions
    data_color = '#F0E68C'       # Light yellow for data
    output_color = '#DDA0DD'     # Light purple for outputs
    cv_color = '#FFE6CC'         # Light orange for CV step
    
    # Title
    ax.text(6, 17.5, 'Random Forest Model Training - Overview', 
            fontsize=18, fontweight='bold', ha='center', color='darkblue')
    
    # Helper functions
    def create_box(x, y, width, height, text, color):
        """Create a simple rectangle box"""
        rect = patches.Rectangle((x-width/2, y-height/2), width, height, 
                                facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=10, ha='center', va='center', 
                weight='bold', wrap=True)
        return rect
    
    def create_diamond(x, y, width, height, text, color):
        """Create a simple diamond for decisions"""
        diamond = patches.RegularPolygon((x, y), 4, radius=width/2, 
                                        orientation=np.pi/4, facecolor=color, 
                                        edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, fontsize=10, ha='center', va='center', 
                weight='bold', wrap=True)
        return diamond
    
    def create_arrow(start_pos, end_pos, label=""):
        """Create a simple arrow with optional label"""
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if label:
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom', 
                    fontsize=9, weight='bold', color='darkred')
    
    # Flowchart elements - High-level overview
    
    # 1. START
    create_box(6, 16, 2, 0.8, 'START', start_color)
    
    # 2. Load Data
    create_box(6, 14.5, 4, 1, 'Load UAM Survey Data\n(Features & Target Variables)', data_color)
    create_arrow((6, 15.6), (6, 15))
    
    # 3. Data Preprocessing
    create_box(6, 13, 4, 1, 'Data Preprocessing:\n• Handle missing values\n• Feature scaling\n• Data validation', data_color)
    create_arrow((6, 14), (6, 13.5))
    
    # 4. Split Data
    create_box(6, 11.5, 4, 1, 'Split Data:\n• Train+Val (80%)\n• Test (20%)\n• Stratified split', data_color)
    create_arrow((6, 12.5), (6, 12))
    
    # 5. Initialize Model
    create_box(6, 10, 4, 1, 'Initialize Random Forest:\n• RandomForestClassifier\n• Set random seed\n• Define pipeline', process_color)
    create_arrow((6, 11), (6, 10.5))
    
    # 6. 10-Fold Cross-Validation
    create_box(6, 8.5, 4, 1, '10-Fold Cross-Validation:\n• Hyperparameter tuning\n• Model evaluation\n• Performance metrics\n(See detailed CV flowchart)', cv_color)
    create_arrow((6, 9.5), (6, 9))
    
    # 7. Get Best Parameters
    create_box(6, 7, 4, 1, 'Extract Best Parameters:\n• Most common hyperparameters\n• Parameter stability analysis\n• Optimal configuration', output_color)
    create_arrow((6, 8), (6, 7.5))
    
    # 8. Train Final Model
    create_box(6, 5.5, 4, 1, 'Train Final Model:\n• Use best parameters\n• Train on full train+val data\n• Calculate training accuracy', process_color)
    create_arrow((6, 6.5), (6, 6))
    
    # 9. Test Set Evaluation
    create_box(6, 4, 4, 1, 'Test Set Evaluation:\n• Predict on test set\n• Calculate final metrics\n• Per-class performance', output_color)
    create_arrow((6, 5), (6, 4.5))
    
    # 10. Feature Importance Analysis
    create_box(6, 2.5, 4, 1, 'Feature Importance Analysis:\n• Calculate mean importance\n• Sort by significance\n• Identify top features', output_color)
    create_arrow((6, 3.5), (6, 3))
    
    # 11. Save Results
    create_box(6, 1, 3, 0.8, 'Save All Results:\n• Model performance\n• Feature importance\n• Predictions', output_color)
    create_arrow((6, 2.1), (6, 1.8))
    
    # 12. END
    create_box(6, -0.5, 2, 0.8, 'END', start_color)
    create_arrow((6, 0.6), (6, 0.3))
    
    # Add a simple legend
    legend_elements = [
        patches.Patch(color=start_color, label='Start/End'),
        patches.Patch(color=data_color, label='Data Operations'),
        patches.Patch(color=process_color, label='Model Operations'),
        patches.Patch(color=cv_color, label='Cross-Validation'),
        patches.Patch(color=output_color, label='Results & Outputs')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add note about detailed CV flowchart
    ax.text(0.5, 0.3, 'Note: Detailed 10-Fold Cross-Validation process', fontsize=12, weight='bold', color='darkblue')
    ax.text(0.5, 0.1, 'is shown in a separate flowchart for clarity', fontsize=10, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('random_forest_overview_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Random Forest Overview flowchart generated successfully!")
    print("Saved as: random_forest_overview_flowchart.png")

if __name__ == "__main__":
    create_rf_overview_flowchart()
