import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_simple_flowchart():
    """Create a simple, easy-to-understand Random Forest flowchart"""
    
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
    
    # Title
    ax.text(6, 17.5, 'Random Forest Model Training Process', 
            fontsize=18, fontweight='bold', ha='center', color='darkblue')
    
    # Helper functions - much simpler
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
    
    # Flowchart elements - much cleaner and simpler
    
    # 1. START
    create_box(6, 16, 2, 0.8, 'START', start_color)
    
    # 2. Load Data
    create_box(6, 14.5, 4, 1, 'Load UAM Survey Data\n(Features & Target)', data_color)
    create_arrow((6, 15.6), (6, 15))
    
    # 3. Split Data
    create_box(6, 13, 4, 1, 'Split Data:\n80% Train/Val, 20% Test', data_color)
    create_arrow((6, 13.5), (6, 13.5))
    
    # 4. Setup Cross-Validation
    create_box(6, 11.5, 4, 1, 'Setup 10-Fold\nCross-Validation', process_color)
    create_arrow((6, 12.5), (6, 12))
    
    # 5. Start CV Loop
    create_diamond(6, 10, 3, 1.2, 'All 10 Folds\nCompleted?', decision_color)
    create_arrow((6, 11), (6, 10.6))
    
    # 6. Process Fold (No path) - Left side
    create_box(2, 8.5, 3, 1, 'Process Current Fold:\n• Split train/val\n• Tune hyperparameters\n• Train model\n• Evaluate', process_color)
    create_arrow((4.5, 9.4), (3.5, 9))
    
    # 7. Store Results - Left side
    create_box(2, 7, 3, 1, 'Store Results:\n• Accuracy, Precision\n• Recall, F1-Score\n• ROC AUC', output_color)
    create_arrow((2, 8), (2, 7.5))
    
    # 8. Loop back - Clean arrow
    create_arrow((2, 6.5), (4.5, 9.4), "Next Fold")
    
    # 9. Post-CV Processing (Yes path) - Right side
    create_box(10, 8.5, 3, 1, 'Post-CV Processing:\n• Average all results\n• Find best parameters\n• Save probabilities', output_color)
    create_arrow((7.5, 9.4), (8.5, 9))
    
    # 10. Train Final Model - Right side
    create_box(10, 7, 3, 1, 'Train Final Model\non Full Train Data', process_color)
    create_arrow((10, 8), (10, 7.5))
    
    # 11. Test Evaluation - Right side
    create_box(10, 5.5, 3, 1, 'Evaluate on Test Set:\n• Calculate all metrics\n• Per-class accuracy', output_color)
    create_arrow((10, 6.5), (10, 6))
    
    # 12. Feature Importance - Center
    create_box(6, 4, 4, 1, 'Analyze Feature Importance:\n• Calculate mean importance\n• Sort and save top features', output_color)
    create_arrow((10, 5), (8, 4.5))
    
    # 13. Save Results - Center
    create_box(6, 2.5, 3, 0.8, 'Save All Results\n& Logs', output_color)
    create_arrow((6, 3.5), (6, 2.9))
    
    # 14. END
    create_box(6, 1, 2, 0.8, 'END', start_color)
    create_arrow((6, 2.1), (6, 1.8))
    
    # Add a simple legend
    legend_elements = [
        patches.Patch(color=start_color, label='Start/End'),
        patches.Patch(color=data_color, label='Data Operations'),
        patches.Patch(color=process_color, label='Model Operations'),
        patches.Patch(color=decision_color, label='Decisions'),
        patches.Patch(color=output_color, label='Results & Outputs')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add key points summary
    ax.text(0.5, 0.3, 'Key Points:', fontsize=12, weight='bold', color='darkblue')
    ax.text(0.5, 0.1, '• 10-fold cross-validation for robust evaluation\n• Hyperparameter tuning with GridSearchCV\n• Comprehensive metrics calculation\n• Feature importance analysis', 
            fontsize=10, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('simple_random_forest_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Simple Random Forest flowchart generated successfully!")
    print("Saved as: simple_random_forest_flowchart.png")

if __name__ == "__main__":
    create_simple_flowchart()
