import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
import random
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the trained model from Part 1
logger.info("Loading trained XGBoost model from Part 1...")
with open(
        "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Trained_Model_XgBoost/xgboost_model_LighterModel.pkl","rb"
    ) as f:
    model_data = pickle.load(f)

final_model = model_data['final_model'] # model
feature_names = model_data['feature_names'] # feature
classes = model_data['classes']
class_names = model_data.get('class_names', {})  # Get class names if available
best_params = model_data['best_params']

logger.info(f"Model loaded successfully. Test accuracy: {model_data['test_acc']:.4f}")
logger.info("Class mapping:")
for class_num, class_name in class_names.items():
    logger.info(f"  {class_num}: {class_name}")

# Load synthetic population data (UAM-unaware data for prediction)
logger.info("Loading processed synthetic population data...")
synthetic_population = pd.read_csv(
    "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/DataPreprocessing_ML.csv" , low_memory=False)
# Sample 1% of the synthetic population data
synthetic_population = synthetic_population.sample(frac=0.01, random_state=42).reset_index(drop=True)


# ==========================================
# 2. INITIALIZE K-MEANS++ WITH 74 VERTIPORTS
# ==========================================
logger.info(
    "Step 2: Initializing k-means++ with 74 vertiports on O/D points from synthetic population data for 1% of the population...")
od_points = np.vstack([
    synthetic_population[['originX', 'originY']].values,
    synthetic_population[['destinationX', 'destinationY']].values
])
kmeans = KMeans(n_clusters=74, init='k-means++', random_state=42, max_iter=1000)
kmeans.fit(od_points)
vertiport_coords = kmeans.cluster_centers_  # This is in meters
logger.info("Step 2 complete: Initial vertiport locations set.")

# =========================
# 3. ITERATIVE OPTIMIZATION
# =========================

# Set car speed and cost
average_car_speed = 418.33  #unit: m/min     ,25.1 km/h, (TomTom- munich: https://www.tomtom.com/traffic-index/munich-traffic/)
cost_per_m_car = 0.00065   #unit:€/m        ,0.65 €/km (Manuscript Number: JTRP-D-24-00632R1)
logger.info(f"Using car speed: {average_car_speed:.2f} m/min, car cost per m: {cost_per_m_car:.6f} €/m")

# --- Centroid history tracking ---
centroid_history = [vertiport_coords.copy()]
convergence_history = []  # Track centroid shifts per iteration
weight_history = []  # Track weights per iteration
uam_prob_history = []  # Track UAM probabilities per iteration
distance_change_history = []  # Track distance matrix changes per iteration
prob_change_history = []  # Track probability changes per iteration

# UAM calculation function, based on assumptions from the literature
vertiport_k = 74
uam_cruise_speed= 5833.33   # unit:m/min     ,350 km/h
uam_cost_m = 0.001  # unit: €/pm,           1 €/pkm
base_fare_uam = 18.4 # unit : €
pre_flight_time = 15 # unit : min
circuity_factor = 1.215 # (Kim et al., 2025) # for car

# --- calculate UAM time and cost  function---
def calculate_uam_time_cost(df, vertiport_coords, car_speed =average_car_speed, car_cost =cost_per_m_car, base_fare= base_fare_uam,
                            uam_speed=uam_cruise_speed, cost_uam_m= uam_cost_m,
                            pre_flight_time=pre_flight_time):
    from scipy.spatial.distance import cdist

    #origin(x,y) and destination (x,y) are in meter
    origins = df[['originX', 'originY']].values
    dests = df[['destinationX', 'destinationY']].values
    
    # Now both origins/dests and vertiport_coords are in meters
    origin_v_idx = np.argmin(cdist(origins, vertiport_coords), axis=1)
    dest_v_idx = np.argmin(cdist(dests, vertiport_coords), axis=1)
    origin_v = vertiport_coords[origin_v_idx]
    dest_v = vertiport_coords[dest_v_idx]

    # first and last mile car distances (in m)
    first_mile_dist = np.linalg.norm(origins - origin_v, axis=1) *  circuity_factor
    last_mile_dist = np.linalg.norm(dests - dest_v, axis=1) *  circuity_factor

    # UAM distance (Euclidean distance in m)
    uam_dist = np.linalg.norm(origin_v - dest_v, axis=1)
    
    # first, last, airborne and total time calculation in min
    first_mile_time = first_mile_dist / car_speed  # unit: min
    last_mile_time = last_mile_dist / car_speed   # unit: min
    airborne_time = uam_dist / uam_speed  # unit: min
    total_time = pre_flight_time + first_mile_time + airborne_time + last_mile_time # unit: min

    #first, last and travel cost calculation in €
    first_mile_cost = first_mile_dist * car_cost
    last_mile_cost = last_mile_dist * car_cost
    uam_travel_cost = base_fare + (cost_uam_m * uam_dist) + first_mile_cost + last_mile_cost
    
    df = df.copy()
    df['travel_time_Uam'] = total_time # min
    df['in_vehicle_time_Uam']= airborne_time # min
    df['waiting_time_Uam']= total_time - airborne_time # min
    df['travel_cost_Uam'] = uam_travel_cost
    df['uam_first_mile'] = first_mile_dist #m
    df['uam_last_mile'] = last_mile_dist #m
    df['uam_air'] = uam_dist #m
    df['uam_origin_vertiport'] = origin_v_idx #vertiport index
    df['uam_dest_vertiport'] = dest_v_idx #vertiport index
    return df

# --- predict mode probabilities function---
def predict_mode_probabilities(df, model,
                               feature_cols):  # features are arranged identically to how they were during training
    X = df[feature_cols]
    return model.predict_proba(X)

# --- Layout similarity check functions ---
def check_distance_matrix_stability(prev_coords, new_coords, threshold=0.01):
    """ Check if the pairwise distance matrix between vertiports is stable
    Returns: (is_stable, max_change)
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Calculate pairwise distance matrices
    prev_distances = squareform(pdist(prev_coords))
    new_distances = squareform(pdist(new_coords))
    
    # Calculate relative change in distances
    relative_change = np.abs(new_distances - prev_distances) / (prev_distances + 1e-8)
    max_change = np.max(relative_change)
    
    return max_change < threshold, max_change

def check_probability_similarity(prev_probs, new_probs, threshold=0.01):
    """ Check if UAM probabilities are stable between iterations
    Returns: (is_stable, max_change)
    """
    # Calculate relative change in probabilities
    relative_change = np.abs(new_probs - prev_probs) / (prev_probs + 1e-8)
    max_change = np.max(relative_change)
    
    return max_change < threshold, max_change


max_iter = 5000
distance_stability_threshold = 0.05  # 5% relative change threshold 
probability_stability_threshold = 0.05  # 5% relative change threshold for probabilities
converged = False
prev_coords = None
feature_cols = feature_names

# Using raw probabilities as weights without normalizing
logger.info("Using raw UAM probabilities as weights")

# Initialize prev_uam_probs for weight smoothing
prev_uam_probs = None

# Initialize min_total_shift for convergence tracking
min_total_shift = float('nan')

for iteration in range(max_iter):
    logger.info(f"Iteration {iteration + 1}...")
    # a. Calculate UAM travel time and cost for each trip
    synthetic_population_with_uam = calculate_uam_time_cost(synthetic_population, vertiport_coords, average_car_speed,
                                                            cost_per_m_car)    # synthetic_population_with_uam is the DataFrame with UAM calculations, only ML features, no UAM calculations 
    synthetic_population_with_uam_full = synthetic_population_with_uam.copy()  # Keep full version for final output

    # b. Predict mode probabilities
    for col in feature_cols:  # Feature Alignment Safeguards
        if col not in synthetic_population_with_uam.columns:
            synthetic_population_with_uam[col] = 0.0
    synthetic_population_with_uam = synthetic_population_with_uam[feature_cols]
    proba = predict_mode_probabilities(synthetic_population_with_uam, final_model, feature_cols)

    # c. Use UAM probability as weights for weighted k-means
    uam_class_idx = None
    for i, cls in enumerate(classes):
        if 'uam' in str(cls).lower() or cls == 4:
            uam_class_idx = i
            break
    if uam_class_idx is None:
        uam_class_idx = len(classes) - 1

    # Log which class is being used for UAM weighting
    uam_class_name = class_names.get(classes[uam_class_idx], f"Class {classes[uam_class_idx]}")
    logger.info(f"Using {uam_class_name} (class {classes[uam_class_idx]}) for UAM probability weighting")
    uam_probs = proba[:, uam_class_idx]

    # Checking for NaN or Infinite Values in UAM Probabilities
    if np.isnan(uam_probs).any():
        logger.error("NaN found in UAM probabilities!")
        raise ValueError("NaN found in UAM probabilities!")
    if not np.isfinite(uam_probs).all():
        logger.error("Infinite value found in UAM probabilities!")
        raise ValueError("Infinite value found in UAM probabilities!")

    # Use raw UAM probabilities as weights for k-means
    weights = np.concatenate([uam_probs, uam_probs])
    
    # Store current UAM probabilities for next iteration
    prev_uam_probs = uam_probs.copy()
    
    # Track weights and probabilities for this iteration
    weight_history.append(weights.copy())
    uam_prob_history.append(uam_probs.copy())

    # Define origins and dests before stacking
    origins = synthetic_population[['originX', 'originY']].values
    dests = synthetic_population[['destinationX', 'destinationY']].values
    od_points_current = np.vstack([origins, dests])

    # f. Perform weighted k-means clustering
    kmeans = KMeans(n_clusters=vertiport_k, init='k-means++', random_state=42, max_iter=1000, 
                    tol=1e-4, n_init=10)  # More stable parameters
    kmeans.fit(od_points_current, sample_weight=weights)  # od_points_current is in meters
    new_coords = kmeans.cluster_centers_  # Output is in meters because input was in meters
    logger.info(f'KMeans finished in {kmeans.n_iter_} iterations this step.')
    # g. Check convergence
    # Robust convergence check using the Hungarian (assignment) algorithm:
    # Computes the minimum total shift between new and previous vertiport coordinates,
    # matching centroids optimally regardless of their order.
    # This prevents false non-convergence due to centroid reordering between iterations.
    if prev_coords is not None:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        cost_matrix = cdist(new_coords, prev_coords) # cost_matrix is in meters
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        min_total_shift = cost_matrix[row_ind, col_ind].sum()  #check
        convergence_history.append(min_total_shift)

        # Reorder new_coords to match prev_coords order
        new_coords_ordered = np.zeros_like(new_coords)
        new_coords_ordered[col_ind] = new_coords[row_ind]

        # NEW: Check distance matrix stability
        distance_stable, distance_change = check_distance_matrix_stability(prev_coords, new_coords_ordered, distance_stability_threshold)
        
        # NEW: Check probability similarity
        prob_stable, prob_change = check_probability_similarity(prev_uam_probs, uam_probs, probability_stability_threshold)
        
        # Track convergence metrics
        distance_change_history.append(distance_change)
        prob_change_history.append(prob_change)
        
        logger.info(f"Iteration {iteration + 1}: Coordinate shift = {min_total_shift:.2f}m, Distance matrix change = {distance_change:.6f} (stable: {distance_stable}), Probability change = {prob_change:.6f} (stable: {prob_stable})")

        # Check for convergence (layout-based + probability-based only)
        # Require at least 3 iterations before allowing convergence
        layout_converged = distance_stable
        probability_converged = prob_stable
        
        if (layout_converged or probability_converged) and iteration >= 2:
            if layout_converged:
                logger.info(f"Converged after {iteration + 1} iterations with distance matrix stability: {distance_change:.6f}")
            if probability_converged:
                logger.info(f"Converged after {iteration + 1} iterations with probability stability: {prob_change:.6f}")
            converged = True
            centroid_history.append(new_coords_ordered.copy())
            break
        prev_coords = new_coords_ordered # prev_coords gets updated  here
        vertiport_coords = new_coords_ordered
        centroid_history.append(vertiport_coords.copy())
    else:
        # First iteration: no reordering needed
        prev_coords = new_coords # prev_coords gets set to meters here
        vertiport_coords = new_coords
        centroid_history.append(vertiport_coords.copy())

if not converged:
    logger.warning(f"Did not converge within {max_iter} iterations. Final shift: {min_total_shift:.6f}")

logger.info("Step 3 complete: Vertiport optimization finished.")

# Save centroid history after optimization
centroid_history = np.array(centroid_history)  # shape: (num_iterations+1, 20, 2)

# Create directory before saving
import os
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Centroid', exist_ok=True)

np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/vertiport_centroid_history.npy',
        centroid_history)

# Save weight and probability histories
weight_history = np.array(weight_history)  # shape: (num_iterations, 2*num_trips)
uam_prob_history = np.array(uam_prob_history)  # shape: (num_iterations, num_trips)
distance_change_history = np.array(distance_change_history)  # shape: (num_iterations,)
prob_change_history = np.array(prob_change_history)  # shape: (num_iterations,)

# Create directory for all history files
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Centroid', exist_ok=True)

np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/weight_history.npy', weight_history)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/uam_prob_history.npy', uam_prob_history)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/distance_change_history.npy', distance_change_history)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/prob_change_history.npy', prob_change_history)

logger.info(f"Weight history saved: {weight_history.shape}")
logger.info(f"UAM probability history saved: {uam_prob_history.shape}")
logger.info(f"Distance change history saved: {distance_change_history.shape}")
logger.info(f"Probability change history saved: {prob_change_history.shape}")

# --- Save final vertiport coordinates (always) ---
import os

centroid_dir = 'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid'
os.makedirs(centroid_dir, exist_ok=True)

# Save as .npy
np.save(os.path.join(centroid_dir, 'optimized_vertiport_coords_final.npy'), vertiport_coords)

# Save as .csv
pd.DataFrame(vertiport_coords, columns=['X', 'Y']).to_csv(
    os.path.join(centroid_dir, 'optimized_vertiport_coords_final.csv'), index=False)

# --- Save convergence history ---
pd.DataFrame({'iteration': list(range(1, len(convergence_history) + 1)), 'shift': convergence_history}).to_csv(
    os.path.join(centroid_dir, 'convergence_history.csv'), index=False)

# --- Save convergence plot ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(convergence_history) + 1), convergence_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Assignment-based vertiport shift')
plt.title('Vertiport Optimization Convergence')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(centroid_dir, 'convergence_plot.png'))
plt.close()
# =========================
# 4. FINAL PREDICTION AND OUTPUT
# =========================
logger.info("Step 4: Final prediction with optimized vertiports and saving results...")
# Use the final results from the last iteration (no need to recalculate)
output = synthetic_population.copy()
for i, cls in enumerate(classes):
    class_name = class_names.get(cls, f"Class_{cls}")
    output[f'prob_mode_{class_name}'] = proba[:, i]  # Use existing proba from last iteration
# Add UAM columns from the existing DataFrame
for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel_time_Uam', 'travel_cost_Uam', 'uam_first_mile',
            'uam_last_mile', 'uam_air']:
    output[col] = synthetic_population_with_uam_full[col]  # Use the FULL DataFrame with UAM calculations

# Save main prediction file with error handling
try:
    os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering', exist_ok=True)
    output_path = '../../../Result/Vertiport_analysis/Probability_clustering/Xgboost_synthetic_population_predictions_raw_weights.csv'
    output.to_csv(output_path, index=False)
    logger.info(f"Main prediction file saved: {output_path}")
except Exception as e:
    logger.error(f"Error saving main prediction file: {e}")

# Save optimized coordinates with error handling
try:
    np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_raw_weights.npy',
            vertiport_coords)
    logger.info("Optimized coordinates saved successfully")
except Exception as e:
    logger.error(f"Error saving optimized coordinates: {e}")

# =========================
# 5. SAVE SUMMARY AND REPORT
# =========================
import csv

report_path = '../../../Result/Vertiport_analysis/Probability_clustering/Weighting/method_report.txt'

# Save summary and report with error handling
try:
    # Create directory for summary file
    os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Weighting', exist_ok=True)

    # Calculate statistics
    mean_uam_prob = float(np.mean(uam_probs))
    std_uam_prob = float(np.std(uam_probs))
    min_uam_prob = float(np.min(uam_probs))
    max_uam_prob = float(np.max(uam_probs))
    final_shift = float(convergence_history[-1]) if convergence_history else float('nan')
    iterations = len(centroid_history) - 1
    
    # Save summary CSV
    summary_path = '../../../Result/Vertiport_analysis/Probability_clustering/Weighting/method_summary.csv'
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Converged', 'Iterations', 'Final_Shift', 'Mean_UAM_Probability', 'Std_UAM_Probability',
                         'Min_UAM_Probability', 'Max_UAM_Probability'])
        writer.writerow([
            'raw_weights', converged, iterations, final_shift, mean_uam_prob, std_uam_prob, min_uam_prob, max_uam_prob
        ])
    logger.info(f"Summary CSV saved: {summary_path}")

    # Save detailed text report
    with open(report_path, 'w') as f:
        f.write(f"VERTIPORT OPTIMIZATION REPORT\n")
        f.write(f"{'=' * 50}\n")
        from datetime import datetime

        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"SUMMARY:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"Method: raw_weights (no normalization)\n")
        f.write(f"Converged: {converged}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Final Shift: {final_shift:.6f}\n")
        f.write(f"Mean UAM Probability: {mean_uam_prob:.4f}\n")
        f.write(f"UAM Probability Std: {std_uam_prob:.4f}\n")
        f.write(f"UAM Probability Range: [{min_uam_prob:.4f}, {max_uam_prob:.4f}]\n")
        f.write(f"\nCONVERGENCE HISTORY (first 20 shown):\n")
        for i, shift in enumerate(convergence_history[:20]):
            f.write(f"  Iter {i + 1}: {shift:.6f}\n")
        if len(convergence_history) > 20:
            f.write(f"  ... ({len(convergence_history) - 20} more)\n")
        f.write(f"\nAll results saved to Result/Vertiport_analysis/Probability_clustering/\n")
    
    logger.info(f"Detailed report saved: {report_path}")
    logger.info("Step 5 complete: Summary and report saved.")
    
except Exception as e:
    logger.error(f"Error saving summary and report: {e}")
logger.info("Results saved with raw UAM probability weights")
print(f"Iterations: {len(centroid_history) - 1}")
print(f"Final shift: {convergence_history[-1] if convergence_history else 'N/A'}") 