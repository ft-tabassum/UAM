import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_simple_vertiport_flowchart():
    """Create a simple, easy-to-understand Vertiport Optimization flowchart"""
    
    # Create figure with better proportions
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Simple, clean colors
    start_color = '#90EE90'      # Light green for start/end
    process_color = '#87CEEB'    # Light blue for processes
    decision_color = '#FFB6C1'   # Light pink for decisions
    data_color = '#F0E68C'       # Light yellow for data
    output_color = '#DDA0DD'     # Light purple for outputs
    
    # Title
    ax.text(5, 15.5, 'Vertiport Optimization Algorithm', 
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
    create_box(5, 14, 2, 0.8, 'START', start_color)
    
    # 2. Load Data & Model
    create_box(5, 12.5, 4, 1, 'Load Data:\n• XGBoost Model\n• Synthetic Population\n• Origin-Destination Points', data_color)
    create_arrow((5, 13.6), (5, 13))
    
    # 3. Initialize Vertiports
    create_box(5, 11, 4, 1, 'Initialize 74 Vertiports\nusing K-means++\non O/D Points', process_color)
    create_arrow((5, 12), (5, 11.5))
    
    # 4. Set Parameters
    create_box(5, 9.5, 4, 1, 'Set UAM Parameters:\n• Speed: 350 km/h\n• Cost: 1 €/km\n• Car: 25.1 km/h, 0.65 €/km', data_color)
    create_arrow((5, 10.5), (5, 10))
    
    # 5. Start Iteration Loop
    create_diamond(5, 8, 3, 1.2, 'All Iterations\nCompleted?', decision_color)
    create_arrow((5, 9), (5, 8.6))
    
    # 6. First Iteration Check
    create_diamond(5, 6.5, 3, 1.2, 'First\nIteration?', decision_color)
    create_arrow((5, 7.4), (5, 7))
    
    # 7. Unweighted Clustering (Yes path)
    create_box(2, 5, 3, 1, 'Unweighted\nK-means\nClustering', process_color)
    create_arrow((3.5, 5.9), (3.5, 5.5))
    
    # 8. Calculate UAM Features (No path)
    create_box(8, 5, 3, 1, 'Calculate UAM\nTime & Cost\nfor all O/D pairs', process_color)
    create_arrow((6.5, 5.9), (6.5, 5.5))
    
    # 9. Predict Probabilities
    create_box(8, 3.5, 3, 1, 'Predict Mode\nProbabilities\nusing XGBoost', process_color)
    create_arrow((8, 4.5), (8, 4))
    
    # 10. Extract Weights
    create_box(8, 2, 3, 1, 'Extract UAM\nProbabilities\nas Weights', process_color)
    create_arrow((8, 3), (8, 2.5))
    
    # 11. Weighted Clustering
    create_box(8, 0.5, 3, 1, 'Weighted K-means\nClustering\n(γ=0.95, α=0.35)', process_color)
    create_arrow((8, 1.5), (8, 1))
    
    # 12. Update Vertiports
    create_box(5, -1, 4, 1, 'Update Vertiport\nCoordinates\nwith New Centroids', process_color)
    create_arrow((2, 4.5), (4, -0.6))
    create_arrow((8, 0), (6, -0.6))
    
    # 13. Check Convergence
    create_diamond(5, -2.5, 3, 1.2, 'Converged?\n(Distance & Probability\nStability)', decision_color)
    create_arrow((5, -1.5), (5, -2.1))
    
    # 14. Continue Loop (No)
    create_box(8, -4, 2.5, 0.8, 'Continue\nLoop', process_color)
    create_arrow((6.5, -2.1), (6.5, -3.6))
    
    # 15. Final Results (Yes)
    create_box(2, -4, 3, 1, 'Save Final\nVertiport\nLocations', output_color)
    create_arrow((3.5, -2.1), (3.5, -3.6))
    
    # 16. END
    create_box(5, -5.5, 2, 0.8, 'END', start_color)
    create_arrow((2, -3.5), (4, -5.1))
    create_arrow((8, -3.6), (6, -5.1))
    
    # Add a simple legend
    legend_elements = [
        patches.Patch(color=start_color, label='Start/End'),
        patches.Patch(color=data_color, label='Data & Parameters'),
        patches.Patch(color=process_color, label='Processing Steps'),
        patches.Patch(color=decision_color, label='Decisions'),
        patches.Patch(color=output_color, label='Results')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add key points summary
    ax.text(0.5, -6.5, 'Key Features:', fontsize=12, weight='bold', color='darkblue')
    ax.text(0.5, -6.8, '• Uses UAM probabilities as weights for clustering\n• Iterative optimization until convergence\n• Combines ML predictions with spatial optimization\n• Balances accessibility and demand', 
            fontsize=10, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('simple_vertiport_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Simple Vertiport Optimization flowchart generated successfully!")
    print("Saved as: simple_vertiport_flowchart.png")

if __name__ == "__main__":
    create_simple_vertiport_flowchart()
