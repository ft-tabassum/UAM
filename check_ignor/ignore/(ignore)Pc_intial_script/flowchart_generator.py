import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_flowchart():
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # Colors
    start_color = '#90EE90'  # Light green
    process_color = '#87CEEB'  # Light blue
    decision_color = '#FFB6C1'  # Light pink
    data_color = '#F0E68C'  # Light yellow
    end_color = '#DDA0DD'  # Light purple
    
    # Font settings
    title_font = {'fontsize': 16, 'fontweight': 'bold', 'ha': 'center'}
    box_font = {'fontsize': 10, 'ha': 'center', 'va': 'center'}
    small_font = {'fontsize': 8, 'ha': 'center', 'va': 'center'}
    
    # Title
    ax.text(5, 23.5, 'Weighted Clustering Vertiport Optimization Algorithm', **title_font)
    
    # Helper function to create rounded rectangle
    def create_box(x, y, width, height, text, color, font_dict=box_font):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, **font_dict)
        return box
    
    # Helper function to create diamond
    def create_diamond(x, y, width, height, text, color, font_dict=box_font):
        diamond = patches.RegularPolygon((x, y), 4, radius=width/2, 
                                        orientation=np.pi/4, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(diamond)
        ax.text(x, y, text, **font_dict)
        return diamond
    
    # Helper function to create connection
    def connect_boxes(box1_pos, box2_pos, arrow_style='->'):
        ax.annotate('', xy=box2_pos, xytext=box1_pos,
                   arrowprops=dict(arrowstyle=arrow_style, lw=2, color='black'))
    
    # Helper function to create text annotation
    def add_text_annotation(x, y, text, font_dict=small_font):
        ax.text(x, y, text, **font_dict)
    
    # 1. START
    start_box = create_box(5, 22, 3, 0.8, 'START', start_color)
    
    # 2. Load Data and Model
    load_box = create_box(5, 20.5, 4, 1, 'Load XGBoost Model\nLoad Synthetic Population Data\n(1% sample)', data_color)
    connect_boxes((5, 21.6), (5, 21))
    
    # 3. Initialize K-means
    init_box = create_box(5, 19, 4, 1, 'Initialize K-means++\n74 vertiports on O/D points', process_color)
    connect_boxes((5, 20), (5, 19.5))
    
    # 4. Set Parameters
    params_box = create_box(5, 17.5, 4, 1.2, 'Set Parameters:\n• UAM cruise speed: 350 km/h\n• UAM cost: 1 €/pkm\n• Car speed: 25.1 km/h\n• Car cost: 0.65 €/km', data_color)
    connect_boxes((5, 18.5), (5, 18.2))
    
    # 5. Initialize Variables
    vars_box = create_box(5, 16, 4, 1, 'Initialize Variables:\n• Centroid history\n• Convergence history\n• Weight history', data_color)
    connect_boxes((5, 16.8), (5, 16.5))
    
    # 6. Start Iteration Loop
    loop_start = create_box(5, 14.5, 3, 0.8, 'Start Iteration Loop\n(max_iter = 5000)', process_color)
    connect_boxes((5, 15.5), (5, 14.9))
    
    # 7. First Iteration Decision
    first_decision = create_diamond(5, 13, 3, 1.5, 'First\nIteration?', decision_color)
    connect_boxes((5, 14.1), (5, 13.75))
    
    # 8. Unweighted K-means (Yes path)
    unweighted_box = create_box(2, 11.5, 3, 1, 'Unweighted\nK-means Clustering', process_color)
    connect_boxes((3.5, 12.25), (3.5, 12))
    
    # 9. Calculate UAM Features (No path)
    uam_calc_box = create_box(8, 11.5, 3, 1, 'Calculate UAM\nTime & Cost', process_color)
    connect_boxes((6.5, 12.25), (6.5, 12))
    
    # 10. Predict Mode Probabilities
    predict_box = create_box(8, 10, 3, 1, 'Predict Mode\nProbabilities\n(XGBoost)', process_color)
    connect_boxes((8, 10.5), (8, 10.5))
    
    # 11. Extract UAM Probabilities
    extract_box = create_box(8, 8.5, 3, 1, 'Extract UAM\nProbabilities\nas Weights', process_color)
    connect_boxes((8, 9.5), (8, 9.5))
    
    # 12. Weighted K-means
    weighted_box = create_box(8, 7, 3, 1, 'Weighted K-means\nClustering\n(γ=0.95, α=0.35)', process_color)
    connect_boxes((8, 8), (8, 7.5))
    
    # 13. Merge paths
    merge_box = create_box(5, 6, 3, 0.8, 'Update Vertiport\nCoordinates', process_color)
    connect_boxes((2, 11), (4, 6.4))
    connect_boxes((8, 6.5), (6, 6.4))
    
    # 14. Convergence Check
    conv_decision = create_diamond(5, 4.5, 3, 1.5, 'Converged?\n(Distance & Probability\nStability)', decision_color)
    connect_boxes((5, 5.6), (5, 5.25))
    
    # 15. Continue Loop (No path)
    continue_box = create_box(8, 3, 2.5, 0.8, 'Continue\nLoop', process_color)
    connect_boxes((6.5, 3.75), (6.5, 3.4))
    
    # 16. Save Results (Yes path)
    save_box = create_box(2, 3, 2.5, 0.8, 'Save Results', process_color)
    connect_boxes((3.5, 3.75), (3.5, 3.4))
    
    # 17. Final Prediction
    final_box = create_box(5, 1.5, 3, 0.8, 'Final Prediction\n& Output', end_color)
    connect_boxes((2, 3.4), (4, 1.9))
    connect_boxes((8, 3.4), (6, 1.9))
    
    # Add detailed annotations
    add_text_annotation(1, 12.5, 'Yes', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    add_text_annotation(9, 12.5, 'No', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    add_text_annotation(1, 4.5, 'Yes', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    add_text_annotation(9, 4.5, 'No', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    
    # Add loop back arrow
    ax.annotate('', xy=(8, 13.5), xytext=(8, 3.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
    add_text_annotation(8.5, 8.5, 'Loop Back', {'fontsize': 8, 'ha': 'left', 'va': 'center', 'color': 'red', 'fontweight': 'bold'})
    
    # Add detailed process descriptions
    add_text_annotation(0.5, 18.5, 'Convergence Criteria:\n• Distance matrix stability < 1%\n• Probability stability < 0.01%', 
                       {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    add_text_annotation(0.5, 16.5, 'Weighted K-means Parameters:\n• γ (gamma): Weight compression (0.95)\n• α (alpha): Damping factor (0.35)\n• tol: Convergence tolerance (1e-3)', 
                       {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    add_text_annotation(0.5, 14.5, 'UAM Features Calculated:\n• Travel time (first mile + air + last mile)\n• Travel cost (base fare + distance cost)\n• Waiting time (pre-flight + ground access)', 
                       {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    add_text_annotation(9.5, 18.5, 'Output Files:\n• Optimized vertiport coordinates\n• Convergence history\n• Final predictions\n• Visualization plots', 
                       {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    add_text_annotation(9.5, 16.5, 'Stability Checks:\n• Hungarian algorithm for centroid matching\n• Relative change in distance matrix\n• Relative change in UAM probabilities', 
                       {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    plt.tight_layout()
    return fig

def create_detailed_flowchart():
    """Create a more detailed flowchart showing the weighted k-means algorithm"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Colors
    process_color = '#87CEEB'
    decision_color = '#FFB6C1'
    data_color = '#F0E68C'
    
    # Title
    ax.text(5, 19.5, 'Detailed Weighted K-means Algorithm', {'fontsize': 14, 'fontweight': 'bold', 'ha': 'center'})
    
    def create_box(x, y, width, height, text, color):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, {'fontsize': 9, 'ha': 'center', 'va': 'center'})
        return box
    
    def create_diamond(x, y, width, height, text, color):
        diamond = patches.RegularPolygon((x, y), 4, radius=width/2, 
                                        orientation=np.pi/4, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(diamond)
        ax.text(x, y, text, {'fontsize': 9, 'ha': 'center', 'va': 'center'})
        return diamond
    
    def connect_boxes(box1_pos, box2_pos):
        ax.annotate('', xy=box2_pos, xytext=box1_pos,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # 1. Initialize centers
    init_box = create_box(5, 18, 3, 0.8, 'Initialize K centers\nrandomly', process_color)
    
    # 2. Start iteration
    iter_box = create_box(5, 16.5, 2.5, 0.8, 'Start Iteration', process_color)
    connect_boxes((5, 17.6), (5, 16.9))
    
    # 3. Calculate distances
    dist_box = create_box(5, 15, 3, 0.8, 'Calculate distances\nfrom points to centers', process_color)
    connect_boxes((5, 16.1), (5, 15.4))
    
    # 4. Assign labels
    assign_box = create_box(5, 13.5, 2.5, 0.8, 'Assign points to\nnearest center', process_color)
    connect_boxes((5, 14.6), (5, 13.9))
    
    # 5. Check convergence
    conv_check = create_diamond(5, 12, 2.5, 1, 'Labels\nchanged?', decision_color)
    connect_boxes((5, 13.1), (5, 12.5))
    
    # 6. Update centers (Yes path)
    update_box = create_box(2, 10.5, 3, 1, 'Update centers:\nWeighted mean\nwith damping', process_color)
    connect_boxes((3.5, 11.5), (3.5, 11))
    
    # 7. Check max shift
    shift_check = create_diamond(2, 9, 2.5, 1, 'Max shift\n< tolerance?', decision_color)
    connect_boxes((2, 10), (2, 9.5))
    
    # 8. Continue (No path)
    continue_box = create_box(2, 7.5, 2, 0.8, 'Continue\niteration', process_color)
    connect_boxes((2, 8.5), (2, 8))
    
    # 9. Return results (Yes path)
    return_box = create_box(5, 7.5, 2.5, 0.8, 'Return final\ncenters & labels', process_color)
    connect_boxes((3.5, 8.5), (4, 7.9))
    
    # Add annotations
    ax.text(0.5, 11.5, 'No', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    ax.text(0.5, 8.5, 'No', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    ax.text(3.5, 8.5, 'Yes', {'fontsize': 10, 'ha': 'center', 'va': 'center', 'fontweight': 'bold'})
    
    # Loop back arrow
    ax.annotate('', xy=(5, 16.1), xytext=(2, 8),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red', linestyle='--'))
    ax.text(2.5, 12, 'Loop Back', {'fontsize': 8, 'ha': 'center', 'va': 'center', 'color': 'red', 'fontweight': 'bold'})
    
    # Add detailed formulas
    ax.text(0.5, 15, 'Weighted Mean Formula:\nμ_w = Σ(w_i * x_i) / Σ(w_i)\n\nDamping Update:\nnew_center = (1-α)*old + α*μ_w', 
            {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    ax.text(7, 15, 'Weight Compression:\nw_compressed = w^γ\n\nParameters:\nγ = 0.95 (compression)\nα = 0.35 (damping)\ntol = 1e-3 (tolerance)', 
            {'fontsize': 8, 'ha': 'left', 'va': 'top'})
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create main flowchart
    fig1 = create_flowchart()
    fig1.savefig('vertiport_optimization_flowchart.png', dpi=300, bbox_inches='tight')
    print("Main flowchart saved as 'vertiport_optimization_flowchart.png'")
    
    # Create detailed weighted k-means flowchart
    fig2 = create_detailed_flowchart()
    fig2.savefig('weighted_kmeans_detailed_flowchart.png', dpi=300, bbox_inches='tight')
    print("Detailed weighted k-means flowchart saved as 'weighted_kmeans_detailed_flowchart.png'")
    
    plt.show()
