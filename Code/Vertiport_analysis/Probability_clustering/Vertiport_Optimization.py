import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
import random
import os
import pickle
from scipy.special import softmax

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create output directories
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Centroid', exist_ok=True)

# Load the trained model from Part 1
logger.info("Loading trained XGBoost model from Part 1...")
with open(
        "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Trained_Model_XgBoost/xgboost_model_LighterModel.pkl","rb"
    ) as f:
    model_data = pickle.load(f)

final_model = model_data['final_model']
feature_names = model_data['feature_names']
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
    "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/synthetic_population_processing.csv")
# Sample 1% of the synthetic population data
synthetic_population = synthetic_population.sample(frac=0.01, random_state=42).reset_index(drop=True)

# =========================
# 2. INITIALIZE K-MEANS++ WITH 74 VERTIPORTS # =========================
logger.info(
    "Step 2: Initializing k-means++ with 74 vertiports on O/D points from synthetic population data for 1% of the population...")
od_points = np.vstack([
    synthetic_population[['originX', 'originY']].values,
    synthetic_population[['destinationX', 'destinationY']].values
])
kmeans = KMeans(n_clusters=74, init='k-means++', random_state=42, max_iter=1000)
kmeans.fit(od_points)
vertiport_coords = kmeans.cluster_centers_
logger.info("Step 2 complete: Initial vertiport locations set.")

# =========================
# 3. ITERATIVE OPTIMIZATION
# =========================
logger.info("Step 3: Iterative vertiport optimization with UAM probability weighting...")
# Filter data to remove unrealistic values (keep only positive distance, time, and cost)
if 'tripLength-km' in synthetic_population.columns and 'travel time_car' in synthetic_population.columns:
    valid = (synthetic_population['travel time_car'] > 0) & (synthetic_population['tripLength-km'] > 0)
    filtered = synthetic_population[valid].copy()
    logger.info(f"Data filtering: {len(filtered)} out of {len(synthetic_population)} trips have valid car data")

if 'TravelCost_Car' in synthetic_population.columns and 'tripLength-km' in synthetic_population.columns:
    valid_cost = (synthetic_population['TravelCost_Car'] > 0) & (synthetic_population['tripLength-km'] > 0)
    filtered_cost = synthetic_population[valid_cost].copy()
    logger.info(f"Data filtering: {len(filtered_cost)} out of {len(synthetic_population)} trips have valid cost data")

# Set car speed and cost
avg_car_speed = 667  # unit: m/min , Default car speed in 40 km/h
car_cost_per_m = 0.00025  # unit:€/m , Default car cost in 0.25 €/km
logger.info(f"Using car speed: {avg_car_speed:.2f} m/min, car cost per m: {car_cost_per_m:.2f} €/m")

# --- Centroid history tracking ---
centroid_history = [vertiport_coords.copy()]
convergence_history = []  # Track centroid shifts per iteration
improvement_threshold = 0.01  # Stop if improvement is less than 1% for 20 iterations
no_improvement_count = 0
patience = 20

# UAM calculation function, based on assumptions from the literature
VERTIPORT_K = 74
UAM_CRUISE_SPEED=5833.33  # unit: meter/min, value:350 km/h
UAM_COST_PER_M = 1000 # unit:pm, value: 1.0 €/pkm
BASE_FARE = 18.4 # unit : €
PRE_FLIGHT_TIME = 15 # unit min


# --- Weight normalization functions ---
def calculate_uam_time_cost(df, vertiport_coords, car_speed, car_cost_m, base_fare=BASE_FARE,
                            uam_speed=UAM_CRUISE_SPEED, uam_cost_m=UAM_COST_PER_M,
                            pre_flight_time=PRE_FLIGHT_TIME):
    from scipy.spatial.distance import cdist
    # Road network factor for car distances (accounts for road network)
    ROAD_NETWORK_FACTOR = 1.4

    origins = df[['originX', 'originY']].values
    dests = df[['destinationX', 'destinationY']].values
    origin_v_idx = np.argmin(cdist(origins, vertiport_coords), axis=1)
    dest_v_idx = np.argmin(cdist(dests, vertiport_coords), axis=1)
    origin_v = vertiport_coords[origin_v_idx]
    dest_v = vertiport_coords[dest_v_idx]
    # Apply road network factor to car distances (first and last mile)
    first_mile_dist = np.linalg.norm(origins - origin_v, axis=1) * ROAD_NETWORK_FACTOR
    last_mile_dist = np.linalg.norm(dests - dest_v, axis=1) * ROAD_NETWORK_FACTOR
    # UAM distance (Euclidean distance)
    uam_dist = np.linalg.norm(origin_v - dest_v,
                              axis=1)  # if the origin and destination is the same, the probability is zero
    first_mile_time = first_mile_dist / car_speed  # unit: min
    last_mile_time = last_mile_dist / car_speed   # unit: min
    airborne_time = uam_dist / uam_speed
    total_time = pre_flight_time + first_mile_time + airborne_time + last_mile_time
    first_mile_cost = first_mile_dist * car_cost_m
    last_mile_cost = last_mile_dist * car_cost_m
    uam_cost = base_fare + (uam_cost_m * uam_dist) + first_mile_cost + last_mile_cost
    df = df.copy()
    df['travel time_Uam'] = total_time
    df['TravelCost_Uam'] = uam_cost
    df['uam_first_mile_m'] = first_mile_dist
    df['uam_last_mile_m'] = last_mile_dist
    df['uam_air_m'] = uam_dist
    df['uam_origin_vertiport'] = origin_v_idx
    df['uam_dest_vertiport'] = dest_v_idx
    return df


def predict_mode_probabilities(df, model,
                               feature_cols):  # features are arranged identically to how they were during training
    X = df[feature_cols]
    return model.predict_proba(X)


max_iter = 3000  #
convergence_threshold = 1.0  # convergence threshold (1km)
converged = False
prev_coords = None
feature_cols = feature_names

# Using raw probabilities as weights without normalizing
logger.info("Using raw UAM probabilities as weights")

min_total_shift = float('nan')

for iteration in range(max_iter):
    logger.info(f"Iteration {iteration + 1}...")
    # a. Calculate UAM travel time and cost for each trip
    synthetic_population_with_uam = calculate_uam_time_cost(synthetic_population, vertiport_coords, avg_car_speed,
                                                            car_cost_per_m)
    # b. Add these UAM features to the synthetic population data (already done in synthetic_population_with_uam)
    # c. Predict mode probabilities
    for col in feature_cols:  # Feature Alignment Safeguards
        if col not in synthetic_population_with_uam.columns:
            synthetic_population_with_uam[col] = 0.0
    synthetic_population_with_uam = synthetic_population_with_uam[feature_cols]
    proba = predict_mode_probabilities(synthetic_population_with_uam, final_model, feature_cols)
    # d. Use UAM probability as weights for weighted k-means
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
    logger.info(
        f"UAM probability stats: min={np.min(uam_probs):.4f}, max={np.max(uam_probs):.4f}, mean={np.mean(uam_probs):.4f}, median={np.median(uam_probs):.4f}, frac_zero={(uam_probs == 0).mean():.4f}")
    if np.isnan(uam_probs).any():
        logger.error("NaN found in UAM probabilities!")
        raise ValueError("NaN found in UAM probabilities!")
    if not np.isfinite(uam_probs).all():
        logger.error("Infinite value found in UAM probabilities!")
        raise ValueError("Infinite value found in UAM probabilities!")

    # --- Softmax normalization per cluster for origins and destinations ---
    from scipy.spatial.distance import cdist
    origins = synthetic_population[['originX', 'originY']].values
    dests = synthetic_population[['destinationX', 'destinationY']].values
    # Assign each point to its nearest vertiport (current centroids)
    origin_assignments = np.argmin(cdist(origins, vertiport_coords), axis=1)
    dest_assignments = np.argmin(cdist(dests, vertiport_coords), axis=1)
    # Softmax-normalize weights within each cluster (origins)
    norm_origin_weights = np.zeros_like(uam_probs)
    for k in range(VERTIPORT_K):
        idx = np.where(origin_assignments == k)[0]
        if len(idx) > 0:
            norm_origin_weights[idx] = softmax(uam_probs[idx])
    # Softmax-normalize weights within each cluster (destinations)
    norm_dest_weights = np.zeros_like(uam_probs)
    for k in range(VERTIPORT_K):
        idx = np.where(dest_assignments == k)[0]
        if len(idx) > 0:
            norm_dest_weights[idx] = softmax(uam_probs[idx])
    # Concatenate for k-means
    weights = np.concatenate([norm_origin_weights, norm_dest_weights])
    if np.isnan(weights).any():
        logger.error("NaN found in weights!")
        raise ValueError("NaN found in weights!")
    if not np.isfinite(weights).all():
        logger.error("Infinite value found in weights!")
        raise ValueError("Infinite value found in weights!")

    od_points_current = np.vstack([origins, dests])
    # f. Perform weighted k-means clustering
    kmeans = KMeans(n_clusters=VERTIPORT_K, init='k-means++', random_state=42, max_iter=1000)
    kmeans.fit(od_points, sample_weight=weights)
    new_coords = kmeans.cluster_centers_
    logger.info(f'KMeans finished in {kmeans.n_iter_} iterations this step.')
    # g. Check convergence
    # Robust convergence check using the Hungarian (assignment) algorithm:
    # Computes the minimum total shift between new and previous vertiport coordinates,
    # matching centroids optimally regardless of their order.
    # This prevents false non-convergence due to centroid reordering between iterations.
    if prev_coords is not None:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        cost_matrix = cdist(new_coords, prev_coords)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # print the shift of each vertiport pairwise
        #print( cost_matrix[row_ind, col_ind])
        min_total_shift = cost_matrix[row_ind, col_ind].sum()
        logger.info(f"Assignment-based vertiport shift: {min_total_shift:.6f}")
        convergence_history.append(min_total_shift)

        # Check for convergence
        if min_total_shift < convergence_threshold:
            logger.info(
                f"Converged after {iteration + 1} iterations with assignment-based shift: {min_total_shift:.6f}")
            converged = True
            centroid_history.append(new_coords.copy())
            break

        # Check for minimal improvement (early stopping)
        if len(convergence_history) > 1:
            improvement = (convergence_history[-2] - convergence_history[-1]) / convergence_history[-2]
            if improvement < improvement_threshold:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping after {iteration + 1} iterations due to minimal improvement")
                    converged = True
                    centroid_history.append(new_coords.copy())
                    break
            else:
                no_improvement_count = 0
    prev_coords = new_coords
    vertiport_coords = new_coords
    centroid_history.append(vertiport_coords.copy())

if not converged:
    logger.warning(f"Did not converge within {max_iter} iterations. Final shift: {min_total_shift:.6f}")

logger.info("Step 3 complete: Vertiport optimization finished.")

# Save centroid history after optimization
centroid_history = np.array(centroid_history)  # shape: (num_iterations+1, 20, 2)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/vertiport_centroid_history.npy',
        centroid_history)

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
# Calculate UAM features (full DataFrame)
synthetic_population_with_uam_full = calculate_uam_time_cost(synthetic_population, vertiport_coords, avg_car_speed,
                                                             car_cost_per_m)
# For prediction, use only model features:
synthetic_population_with_uam = synthetic_population_with_uam_full.copy()
for col in feature_cols:  # Feature Alignment Safeguards
    if col not in synthetic_population_with_uam.columns:
        synthetic_population_with_uam[col] = 0.0
synthetic_population_with_uam = synthetic_population_with_uam[feature_cols]
proba = predict_mode_probabilities(synthetic_population_with_uam, final_model, feature_cols)
output = synthetic_population.copy()
for i, cls in enumerate(classes):
    class_name = class_names.get(cls, f"Class_{cls}")
    output[f'prob_mode_{class_name}'] = proba[:, i]
# Add UAM columns from the full DataFrame
for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel time_Uam', 'TravelCost_Uam', 'uam_first_mile_m',
            'uam_last_mile_m', 'uam_air_m']:
    output[col] = synthetic_population_with_uam_full[col]
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering', exist_ok=True)
output.to_csv(
    '../../../Result/Vertiport_analysis/Probability_clustering/Xgboost_synthetic_population_predictions_raw_weights.csv',
    index=False)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_raw_weights.npy',
        vertiport_coords)

# =========================
# 5. SAVE SUMMARY AND REPORT
# =========================
import csv

report_path = '../../../Result/Vertiport_analysis/Probability_clustering/Weighting/method_report.txt'

# Save summary CSV
mean_uam_prob = float(np.mean(uam_probs))
std_uam_prob = float(np.std(uam_probs))
min_uam_prob = float(np.min(uam_probs))
max_uam_prob = float(np.max(uam_probs))
final_shift = float(convergence_history[-1]) if convergence_history else float('nan')
iterations = len(centroid_history) - 1
summary_path = '/Result/Vertiport_analysis/Probability_clustering/Weighting/method_summary.csv'
with open(summary_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Method', 'Converged', 'Iterations', 'Final_Shift', 'Mean_UAM_Probability', 'Std_UAM_Probability',
                     'Min_UAM_Probability', 'Max_UAM_Probability'])
    writer.writerow([
        'raw_weights', converged, iterations, final_shift, mean_uam_prob, std_uam_prob, min_uam_prob, max_uam_prob
    ])

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

logger.info("Step 5 complete: Summary and report saved.")
logger.info("Results saved with raw UAM probability weights") 