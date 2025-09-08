import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_simple_flowchart():
    """Create a simple flowchart for the vertiport optimization algorithm"""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Colors
    start_color = '#90EE90'  # Light green
    process_color = '#87CEEB'  # Light blue
    decision_color = '#FFB6C1'  # Light pink
    data_color = '#F0E68C'  # Light yellow
    end_color = '#DDA0DD'  # Light purple
    
    # Title
    ax.text(5, 17.5, 'Vertiport Optimization Algorithm Flowchart', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Helper functions
    def create_rectangle(x, y, width, height, text, color):
        rect = patches.Rectangle((x-width/2, y-height/2), width, height, 
                                facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=9, ha='center', va='center')
        return rect
    
    def create_diamond(x, y, width, height, text, color):
        diamond = patches.RegularPolygon((x, y), 4, radius=width/2, 
                                        orientation=np.pi/4, facecolor=color, 
                                        edgecolor='black', linewidth=1)
        ax.add_patch(diamond)
        ax.text(x, y, text, fontsize=9, ha='center', va='center')
        return diamond
    
    def create_arrow(start_pos, end_pos):
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Flowchart elements
    # 1. Start
    create_rectangle(5, 16, 2, 0.8, 'START', start_color)
    
    # 2. Load data
    create_rectangle(5, 14.5, 4, 1, 'Load XGBoost Model\nLoad Synthetic Population\n(1% sample)', data_color)
    create_arrow((5, 15.6), (5, 15))
    
    # 3. Initialize
    create_rectangle(5, 13, 4, 1, 'Initialize K-means++\n74 vertiports on O/D points', process_color)
    create_arrow((5, 14), (5, 13.5))
    
    # 4. Set parameters
    create_rectangle(5, 11.5, 4, 1, 'Set UAM Parameters:\nSpeed, Cost, etc.', data_color)
    create_arrow((5, 12.5), (5, 12))
    
    # 5. Start iteration
    create_rectangle(5, 10, 3, 0.8, 'Start Iteration Loop', process_color)
    create_arrow((5, 11), (5, 10.4))
    
    # 6. First iteration check_ignor
    create_diamond(5, 8.5, 3, 1.2, 'First\nIteration?', decision_color)
    create_arrow((5, 9.6), (5, 9.1))
    
    # 7. Unweighted path (Yes)
    create_rectangle(2, 7, 3, 1, 'Unweighted\nK-means', process_color)
    create_arrow((3.5, 7.9), (3.5, 7.5))
    
    # 8. Weighted path (No)
    create_rectangle(8, 7, 3, 1, 'Calculate UAM\nTime & Cost', process_color)
    create_arrow((6.5, 7.9), (6.5, 7.5))
    
    # 9. Predict probabilities
    create_rectangle(8, 5.5, 3, 1, 'Predict Mode\nProbabilities', process_color)
    create_arrow((8, 6.5), (8, 6))
    
    # 10. Extract weights
    create_rectangle(8, 4, 3, 1, 'Extract UAM\nProbabilities\nas Weights', process_color)
    create_arrow((8, 5), (8, 4.5))
    
    # 11. Weighted clustering
    create_rectangle(8, 2.5, 3, 1, 'Weighted K-means\n(γ=0.95, α=0.35)', process_color)
    create_arrow((8, 3.5), (8, 3))
    
    # 12. Update coordinates
    create_rectangle(5, 1.5, 3, 0.8, 'Update Vertiport\nCoordinates', process_color)
    create_arrow((2, 6.5), (4, 1.9))
    create_arrow((8, 2), (6, 1.9))
    
    # 13. Convergence check_ignor
    create_diamond(5, 0, 3, 1.2, 'Converged?\n(Distance & Probability\nStability)', decision_color)
    create_arrow((5, 1.1), (5, 0.6))
    
    # 14. Continue loop (No)
    create_rectangle(8, -1, 2.5, 0.8, 'Continue\nLoop', process_color)
    create_arrow((6.5, 0.6), (6.5, -0.6))
    
    # 15. Save results (Yes)
    create_rectangle(2, -1, 2.5, 0.8, 'Save Results', process_color)
    create_arrow((3.5, 0.6), (3.5, -0.6))
    
    # 16. Final output
    create_rectangle(5, -2.5, 3, 0.8, 'Final Prediction\n& Output', end_color)
    create_arrow((2, -0.6), (4, -2.1))
    create_arrow((8, -0.6), (6, -2.1))
    
    # Loop back arrow
    ax.annotate('', xy=(8, 8.5), xytext=(8, -0.6),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
    ax.text(8.5, 4, 'Loop Back', fontsize=8, ha='left', va='center', 
            color='red', fontweight='bold')
    
    # Add labels
    ax.text(1, 8.5, 'Yes', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(9, 8.5, 'No', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(1, 0, 'Yes', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(9, 0, 'No', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Add annotations
    ax.text(0.5, 15, 'Convergence Criteria:\n• Distance matrix stability < 1%\n• Probability stability < 0.01%', 
            fontsize=8, ha='left', va='top')
    
    ax.text(0.5, 12, 'Parameters:\n• γ (gamma): Weight compression (0.95)\n• α (alpha): Damping factor (0.35)\n• tol: Convergence tolerance (1e-3)', 
            fontsize=8, ha='left', va='top')
    
    ax.text(9.5, 15, 'Output Files:\n• Optimized vertiport coordinates\n• Convergence history\n• Final predictions\n• Visualization plots', 
            fontsize=8, ha='left', va='top')
    
    ax.text(9.5, 12, 'Stability Checks:\n• Hungarian algorithm for centroid matching\n• Relative change in distance matrix\n• Relative change in UAM probabilities', 
            fontsize=8, ha='left', va='top')
    
    plt.tight_layout()
    return fig

def create_algorithm_summary():
    """Create a summary of the algorithm steps"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Algorithm Summary', fontsize=14, fontweight='bold', ha='center')
    
    # Algorithm steps
    steps = [
        "1. Load pre-trained XGBoost model and synthetic population data (1% sample)",
        "2. Initialize 74 vertiports using K-means++ on origin-destination points",
        "3. Set UAM parameters (speed: 350 km/h, cost: 1 €/pkm, etc.)",
        "4. Start iterative optimization loop (max 5000 iterations):",
        "   a. First iteration: Use unweighted K-means clustering",
        "   b. Subsequent iterations:",
        "      - Calculate UAM travel time and cost for each trip",
        "      - Predict mode choice probabilities using XGBoost",
        "      - Extract UAM probabilities as weights for clustering",
        "      - Apply weighted K-means with compression (γ=0.95) and damping (α=0.35)",
        "      - Update vertiport coordinates",
        "5. Check convergence using Hungarian algorithm and stability criteria",
        "6. Save optimized vertiport locations and final predictions"
    ]
    
    y_pos = 8.5
    for i, step in enumerate(steps):
        if step.startswith("   "):
            # Sub-step
            ax.text(0.5, y_pos, step, fontsize=9, ha='left', va='top', 
                   style='italic', color='blue')
        else:
            # Main step
            ax.text(0.5, y_pos, step, fontsize=10, ha='left', va='top', 
                   fontweight='bold')
        y_pos -= 0.6
    
    # Key features box
    ax.text(0.5, 2, 'Key Features:', fontsize=11, fontweight='bold', ha='left', va='top')
    features = [
        "• Uses UAM probabilities as weights for clustering",
        "• Implements weight compression to handle extreme values",
        "• Applies damping factor to prevent large centroid jumps",
        "• Uses Hungarian algorithm for optimal centroid matching",
        "• Checks both distance matrix and probability stability",
        "• Generates comprehensive visualizations and reports"
    ]
    
    y_pos = 1.5
    for feature in features:
        ax.text(0.5, y_pos, feature, fontsize=9, ha='left', va='top')
        y_pos -= 0.4
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create simple flowchart
    fig1 = create_simple_flowchart()
    fig1.savefig('vertiport_optimization_simple_flowchart.png', dpi=300, bbox_inches='tight')
    print("Simple flowchart saved as 'vertiport_optimization_simple_flowchart.png'")
    
    # Create algorithm summary
    fig2 = create_algorithm_summary()
    fig2.savefig('algorithm_summary.png', dpi=300, bbox_inches='tight')
    print("Algorithm summary saved as 'algorithm_summary.png'")
    
    plt.show()
