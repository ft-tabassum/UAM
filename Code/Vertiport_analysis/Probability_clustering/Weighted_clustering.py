import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import random
import joblib
import os
import matplotlib.pyplot as plt
from scipy import stats

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# Custom imputer that preserves feature names (required for loading pickle file)
class FeaturePreservingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='constant', fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fill missing values while preserving DataFrame structure
        X_filled = X.fillna(self.fill_value)
        return X_filled


# Load the trained model from Part 1 using joblib
logger.info("Loading trained LightGBM model from Part 1...")
model_data = joblib.load(
    "D:/Thesis/UAM/Result/Vertiport_analysis/Model_LightGBM/Trained_Model_LightGBM/lightgbm_model_LighterModel.pkl")

final_model = model_data['final_model']  # model
feature_names = model_data['feature_names']  # feature
classes = model_data['classes']
class_names = model_data.get('class_names', {})  # class names
best_params = model_data['best_params']  # parameters

logger.info(f"Model loaded successfully. Test accuracy: {model_data['test_acc']:.4f}")
logger.info("Class mapping:")
for class_num, class_name in class_names.items():
    logger.info(f"  {class_num}: {class_name}")

# Load synthetic population data (UAM-unaware data for prediction)
logger.info("Loading processed synthetic population data...")
# Load full synthetic population data
synthetic_population = pd.read_csv(
    "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/DataPreprocessing_ML.csv",
    low_memory=False)
# Sample a larger percentage for better representation
synthetic_population = synthetic_population.sample(frac=0.1, random_state=42).reset_index(drop=True)  # 10%

# 2. INITIALIZE K-MEANS++ WITH 74 VERTIPORTS
logger.info(
    "Step 2: Initializing k-means++ with 74 vertiports on O/D points from synthetic population data...")
od_points = np.vstack([
    synthetic_population[['originX', 'originY']].values,
    synthetic_population[['destinationX', 'destinationY']].values])
kmeans = KMeans(n_clusters=74, init='k-means++', random_state=42, max_iter=1000)
kmeans.fit(od_points)
vertiport_coords = kmeans.cluster_centers_  # This is in meters
logger.info("Step 2 complete: Initial vertiport locations set.")

# 3. ITERATIVE OPTIMIZATION
# UAM travel time and travel cost calculation value, based on assumptions from the literature
vertiport_k = 74  # centriod        (Guo et al., 2025)
uam_cruise_speed = 4166.67  # unit:m/min,     250 km/h
uam_cost_m = 0.005  # unit: €/m,     5 €/km
pre_flight_time = 15  # unit : min      (Rothfeld, 2021)
average_car_speed = 418.33  # unit: m/min,    25.1 km/h (TomTom- munich: https://www.tomtom.com/traffic-index/munich-traffic/)
cost_per_m_car = 0.00065  # unit:€/m,       0.65 €/km (Manuscript Number: JTRP-D-24-00632R1)
circuity_factor = 1.215  # for car         (Kim et al., 2025)

# Centroid history tracking
centroid_history = [vertiport_coords.copy()]
convergence_history = []  # Track centroid shifts per iteration
weight_history = []  # Track weights per iteration
uam_prob_history = []  # Track UAM probabilities per iteration
distance_change_history = []  # Track distance matrix changes per iteration
prob_change_history = []  # Track probability changes per iteration


# function to calculate UAM time and travel cost
def calculate_uam_time_cost(df, vertiport_coords, car_speed=average_car_speed, car_cost=cost_per_m_car,
                            uam_speed=uam_cruise_speed, cost_uam_m=uam_cost_m,
                            pre_flight_time=pre_flight_time):
    from scipy.spatial.distance import cdist

    # origin(x,y) and destination (x,y) are in meter
    origins = df[['originX', 'originY']].values
    dests = df[['destinationX', 'destinationY']].values

    # both origins/dests and vertiport_coords are in meters
    origin_v_idx = np.argmin(cdist(origins, vertiport_coords), axis=1)
    dest_v_idx = np.argmin(cdist(dests, vertiport_coords), axis=1)
    origin_v = vertiport_coords[origin_v_idx]
    dest_v = vertiport_coords[dest_v_idx]

    # first and last mile car distances (in m)
    first_mile_dist = np.linalg.norm(origins - origin_v, axis=1) * circuity_factor
    last_mile_dist = np.linalg.norm(dests - dest_v, axis=1) * circuity_factor

    # UAM distance (Euclidean distance in m)
    uam_dist = np.linalg.norm(origin_v - dest_v, axis=1)

    # first, last, airborne and total time calculation in min
    first_mile_time = first_mile_dist / car_speed  # unit: min
    last_mile_time = last_mile_dist / car_speed  # unit: min
    airborne_time = uam_dist / uam_speed  # unit: min
    total_time = pre_flight_time + first_mile_time + airborne_time + last_mile_time  # unit: min

    # first, last and travel cost calculation in €
    first_mile_cost = first_mile_dist * car_cost
    last_mile_cost = last_mile_dist * car_cost
    uam_travel_cost = (cost_uam_m * uam_dist) + first_mile_cost + last_mile_cost

    df = df.copy()
    # Create features with the correct names that the model expects
    df['AFT_TT'] = total_time  # min - UAM travel time
    df['AFT_CO'] = uam_travel_cost  # € - UAM travel cost
    
    # Also keep the original names for compatibility
    df['travel_time_Uam'] = total_time  # min
    df['in_vehicle_time_Uam'] = airborne_time  # min
    df['waiting_time_Uam'] = total_time - airborne_time  # min
    df['travel_cost_Uam'] = uam_travel_cost
    df['uam_first_mile'] = first_mile_dist  # m
    df['uam_last_mile'] = last_mile_dist  # m
    df['uam_air'] = uam_dist  # m
    df['uam_origin_vertiport'] = origin_v_idx  # vertiport index
    df['uam_dest_vertiport'] = dest_v_idx  # vertiport index
    return df


# predict mode probabilities function
def predict_mode_probabilities(df, model,
                               feature_cols):  # features are arranged identically to how they were during training
    X = df[feature_cols]
    return model.predict_proba(X)


# functions for Layout similarity check: the stability of centroid positions (distance matrix)
def check_distance_matrix_stability(prev_coords, new_coords, threshold=0.01):
    """ Check if the pairwise distance matrix between vertiports is stable """
    from scipy.spatial.distance import pdist, squareform

    # Calculate pairwise distance matrices
    prev_distances = squareform(pdist(prev_coords))
    new_distances = squareform(pdist(new_coords))

    # Calculate relative change in distances
    relative_change = np.abs(new_distances - prev_distances) / (
                prev_distances + 1e-8)  # 1e-8 is a small constant to avoid division by zero
    max_change = np.max(relative_change)

    return max_change < threshold, max_change


# function for probability similarity check
def check_probability_similarity(prev_probs, new_probs, threshold=0.05):
    """ Check if UAM probabilities are stable between iterations"""

    # Calculate relative change in probabilities
    relative_change = np.abs(new_probs - prev_probs) / (prev_probs + 1e-8)
    max_change = np.max(relative_change)

    return max_change < threshold, max_change


max_iter = 5000
distance_stability_threshold = 0.05  # 5% relative change threshold
probability_stability_threshold = 0.005  # 0.5% relative change threshold for probabilities
converged = False
prev_coords = None
feature_cols = feature_names

# Initialize min_total_shift for convergence tracking
min_total_shift = float('nan')

# Define centroid directory for saving visualizations
centroid_dir = 'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid'
os.makedirs(centroid_dir, exist_ok=True)


# Visualization function to plot centroid and demand
def plot_centroids_and_demand(centroids, origins, destinations, iteration, save_path):
    """Plot vertiport centroids and demand points"""
    plt.figure(figsize=(12, 10))

    # Plot demand points
    plt.scatter(origins[:, 0], origins[:, 1], c='lightblue', s=0.2, alpha=0.6, label='Origins', marker='o')
    plt.scatter(destinations[:, 0], destinations[:, 1], c='lightgreen', s=0.2, alpha=0.6, label='Destinations',
                marker='s')

    # Plot vertiport centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.8, label='Vertiports', marker='^',
                edgecolors='black', linewidth=1)

    # Add vertiport numbers
    for i, (x, y) in enumerate(centroids):
        plt.annotate(f'V{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                     fontsize=8, fontweight='bold', color='darkred')

    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title(f'Vertiport Locations and Demand Points - Iteration {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Weighted K-Means Implementation with probability-weight compression (gamma) and damping factor for centroid updates (alpha)
def weighted_kmeans(X, w, K, gamma=0.95, alpha=0.50, max_iter=1000, tol=1e-3,
                    random_state=None):  # Parameters- X : Data points; random_state : Random seed

    rng = np.random.default_rng(random_state)
    eps = 1e-12

    # compress weights to make weight distribution more uniform :reduces extreme values (bimodal distribution of weights)
    w = w ** gamma  # w : Weights for each data point

    # initialize centers
    init_idx = rng.choice(len(X), K, replace=False)
    centers = X[init_idx].copy()  # centers : Final cluster centers

    history = {'max_shift': [], 'n_changed': [],
               'inertia': []}  # history : Tracking info: 'max_shift', 'n_changed', 'inertia'
    labels = np.full(len(X), -1)  # labels : Cluster labels

    for it in range(max_iter):
        # assign labels
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        # track changes
        n_changed = np.sum(new_labels != labels)
        labels = new_labels

        # compute inertia (weighted sum of squared distances)
        inertia = np.sum(w * np.min(dists ** 2, axis=1))

        # update centers with damping
        max_shift = 0.0  # Used to determine if the algorithm has converged- if all centroids move less than the tolerance threshold, the algorithm stops.
        for k in range(K):  # Iterates through each of the K clusters (k = 0, 1, 2, ..., K-1)
            mask = labels == k  # Find Points Belonging to Cluster k

            if np.any(
                    mask):  # Check if Cluster Has Any Points- np.any(mask) returns True if at least one point belongs to cluster k
                wk = w[mask]  # Weights of points in cluster k
                Xk = X[mask]  # Coordinates of points in cluster k
                mu_w = (wk[:, None] * Xk).sum(axis=0) / (
                            wk.sum() + eps)  # Calculate Weighted Mean (Centroid); eps: Small constant to prevent division by zero
                new_center = (1 - alpha) * centers[k] + alpha * mu_w  # Prevents large jumps in centroid positions
                shift = np.linalg.norm(new_center - centers[k])
                centers[k] = new_center
                max_shift = max(max_shift, shift)
            else:
                # reinitialize empty cluster - If False, the cluster is empty
                centers[k] = X[rng.integers(len(X))]

        # store history
        history['max_shift'].append(max_shift)
        history['n_changed'].append(n_changed)
        history['inertia'].append(inertia)

        # check convergence
        if max_shift < tol:  # tol : Convergence tolerance (max centroid shift)
            break

    return labels, centers, history

for iteration in range(max_iter):
    logger.info(f"Iteration {iteration + 1}...")

    # Define origins and dests before stacking
    origins = synthetic_population[['originX', 'originY']].values
    dests = synthetic_population[['destinationX', 'destinationY']].values
    od_points_current = np.vstack([origins, dests])

    if iteration == 0:
        # FIRST ITERATION: Unweighted K-means
        logger.info("First iteration: Using unweighted K-means clustering")

        # K-means without weights
        kmeans = KMeans(n_clusters=vertiport_k, init='k-means++', random_state=42, max_iter=1000)
        kmeans.fit(od_points_current)  # NO sample_weight
        new_coords = kmeans.cluster_centers_

        logger.info(f'Unweighted KMeans finished in {kmeans.n_iter_} iterations')

        # No convergence check needed for first iteration
        prev_coords = new_coords
        vertiport_coords = new_coords
        centroid_history.append(vertiport_coords.copy())

        # Save intermediate visualization for first iteration
        intermediate_plot_path = os.path.join(centroid_dir, f'centroids_iteration_{iteration + 1}_unweighted.png')
        plot_centroids_and_demand(
            new_coords,
            origins,
            dests,
            iteration + 1,
            intermediate_plot_path
        )
        logger.info(
            f"First iteration (unweighted) visualization saved: centroids_iteration_{iteration + 1}_unweighted.png")

    else:
        # SUBSEQUENT ITERATIONS: Weighted K-means based on UAM probabilities
        logger.info(f"Iteration {iteration + 1}: Using weighted K-means based on UAM probabilities")

        # a. Calculate UAM travel time and cost for each trip
        synthetic_population_with_uam = calculate_uam_time_cost(synthetic_population, vertiport_coords,
                                                                average_car_speed,
                                                                cost_per_m_car)  # synthetic_population_with_uam is the DataFrame with UAM calculations, only ML features, no UAM calculations
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

        # d. Log which class is being used for UAM weighting
        uam_class_name = class_names.get(classes[uam_class_idx], f"Class {classes[uam_class_idx]}")
        logger.info(f"Using {uam_class_name} (class {classes[uam_class_idx]}) for UAM probability weighting")
        uam_probs = proba[:, uam_class_idx]

        # e. Checking for NaN or Infinite Values in UAM Probabilities
        if np.isnan(uam_probs).any():
            logger.error("NaN found in UAM probabilities!")
            raise ValueError("NaN found in UAM probabilities!")
        if not np.isfinite(uam_probs).all():
            logger.error("Infinite value found in UAM probabilities!")
            raise ValueError("Infinite value found in UAM probabilities!")

        # i. Use raw UAM probabilities as weights (gamma is applied inside weighted_kmeans)
        weights = np.concatenate([uam_probs, uam_probs])

        # j. Store current UAM probabilities for next iteration
        prev_uam_probs = uam_probs.copy()

        # k. Track weights and probabilities for this iteration (will be updated after feedback loop)
        weight_history.append(weights.copy())
        uam_prob_history.append(uam_probs.copy())

        # l. Define parameters for weighted k-means clustering
        gamma = 0.95
        alpha = 0.50
        # Since centroids shift a lot and do not converge (converge plot)  → alpha = 0.35;
        # means 65% old position + 35% new weighted mean
        tol = 1e-3  # Convergence tolerance in meters

        # weighted k-means clustering calculation
        labels, new_coords, kmeans_history = weighted_kmeans(
            X=od_points_current,
            w=weights,
            K=vertiport_k,
            gamma=gamma,
            alpha=alpha,
            max_iter=1000,
            tol=tol,
            random_state=42
        )

        logger.info(f'weighted KMeans finished in {len(kmeans_history["max_shift"])} iterations')
        logger.info(
            f'Final max shift: {kmeans_history["max_shift"][-1]:.6f}, Final inertia: {kmeans_history["inertia"][-1]:.2f}')

        # Track KMeans performance metrics
        if 'kmeans_iterations' not in locals():
            kmeans_iterations = []
            kmeans_max_shifts = []
            kmeans_inertias = []

        kmeans_iterations.append(len(kmeans_history["max_shift"]))
        kmeans_max_shifts.append(kmeans_history["max_shift"][-1])
        kmeans_inertias.append(kmeans_history["inertia"][-1])

        # FEEDBACK LOOP: Recalculate UAM features and probabilities with new vertiport coordinates
        logger.info("Recalculating UAM features and probabilities with new vertiport coordinates...")

        # Update vertiport coordinates to new coordinates from weighted K-means
        vertiport_coords = new_coords.copy()

        # Recalculate UAM features with new vertiport coordinates
        synthetic_population_with_uam_updated = calculate_uam_time_cost(synthetic_population, vertiport_coords,
                                                                        average_car_speed, cost_per_m_car)

        # Get updated probabilities with new UAM features
        for col in feature_cols:
            if col not in synthetic_population_with_uam_updated.columns:
                synthetic_population_with_uam_updated[col] = 0.0
        synthetic_population_with_uam_updated = synthetic_population_with_uam_updated[feature_cols]
        proba_updated = predict_mode_probabilities(synthetic_population_with_uam_updated, final_model, feature_cols)

        # Get updated UAM probabilities
        uam_probs_updated = proba_updated[:, uam_class_idx]

        # Check for NaN or Infinite Values in updated UAM Probabilities
        if np.isnan(uam_probs_updated).any():
            logger.error("NaN found in updated UAM probabilities!")
            raise ValueError("NaN found in updated UAM probabilities!")
        if not np.isfinite(uam_probs_updated).all():
            logger.error("Infinite value found in updated UAM probabilities!")
            raise ValueError("Infinite value found in updated UAM probabilities!")

        # Log probability changes
        prob_change_this_iteration = np.abs(uam_probs_updated - uam_probs).mean()
        logger.info(f"UAM probability change this iteration: {prob_change_this_iteration:.6f}")

        # Update probabilities for next iteration
        uam_probs = uam_probs_updated.copy()

        # Update weight history with new probabilities
        weights_updated = np.concatenate([uam_probs, uam_probs])
        weight_history[-1] = weights_updated.copy()  # Update the last entry
        uam_prob_history[-1] = uam_probs.copy()  # Update the last entry

        # g. Check convergence:  using the Hungarian (assignment) algorithm: Computes the mean shift between new and previous vertiport coordinates, matching centroids optimally regardless of their order. This prevents false non-convergence due to centroid reordering between iterations.
        if prev_coords is not None:
            from scipy.optimize import linear_sum_assignment
            from scipy.spatial.distance import cdist

            cost_matrix = cdist(new_coords, prev_coords)  # Distance matrix,  is in meters
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  # optimal matching
            min_total_shift = cost_matrix[row_ind, col_ind].mean()  # Average shift per vertiport
            convergence_history.append(min_total_shift)

            # Reorder new_coords to match prev_coords order
            new_coords_ordered = np.zeros_like(new_coords)
            new_coords_ordered[col_ind] = new_coords[row_ind]

            # Check distance matrix stability
            distance_stable, distance_change = check_distance_matrix_stability(prev_coords, new_coords_ordered,
                                                                               distance_stability_threshold)

            # Check probability similarity using updated probabilities
            prob_stable, prob_change = check_probability_similarity(prev_uam_probs, uam_probs,
                                                                    probability_stability_threshold)

            # Track convergence metrics
            distance_change_history.append(distance_change)
            prob_change_history.append(prob_change)

            logger.info(
                f"Iteration {iteration + 1}: Coordinate shift = {min_total_shift:.2f}m, Distance matrix change = {distance_change:.6f} (stable: {distance_stable}), Probability change = {prob_change:.6f} (stable: {prob_stable})")

            # Save intermediate visualization every 5 iterations
            if (iteration + 1) % 5 == 0 or iteration < 5:
                intermediate_plot_path = os.path.join(centroid_dir, f'centroids_iteration_{iteration + 1}.png')
                plot_centroids_and_demand(
                    new_coords_ordered,
                    origins,
                    dests,
                    iteration + 1,
                    intermediate_plot_path
                )
                logger.info(f"Intermediate visualization saved: centroids_iteration_{iteration + 1}.png")

            # Check for convergence
            layout_converged = distance_stable
            probability_converged = prob_stable

            # Both layout and probability stable
            if (layout_converged and probability_converged) and iteration >= 2:
                logger.info(
                    f"Converged after {iteration + 1} iterations with BOTH distance matrix stability: {distance_change:.6f} AND probability stability: {prob_change:.6f}")
                converged = True
                centroid_history.append(new_coords_ordered.copy())
                break

            prev_coords = new_coords_ordered  # prev_coords gets updated  here
            vertiport_coords = new_coords_ordered
            centroid_history.append(vertiport_coords.copy())
        else:
            # This should not happen in weighted iterations, but just in case
            prev_coords = new_coords
            vertiport_coords = new_coords
            centroid_history.append(vertiport_coords.copy())

if not converged:
    logger.warning(f"Did not converge within {max_iter} iterations. Final shift: {min_total_shift:.6f}")

logger.info("Step 3 complete: Vertiport optimization finished.")

# Save centroid history after optimization
centroid_history = np.array(centroid_history)  # shape: (num_iterations+1, 20, 2)

# Create directory before saving
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Centroid', exist_ok=True)

np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/vertiport_centroid_history.npy',
        centroid_history)

# Save weight and probability histories
weight_history = np.array(weight_history)  # shape: (num_iterations, 2*num_trips)
uam_prob_history = np.array(uam_prob_history)  # shape: (num_iterations, num_trips)
distance_change_history = np.array(distance_change_history)  # shape: (num_iterations,)
prob_change_history = np.array(prob_change_history)  # shape: (num_iterations,)

# Create directory & save all history files
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Centroid', exist_ok=True)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/weight_history.npy', weight_history)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/uam_prob_history.npy', uam_prob_history)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/distance_change_history.npy',
        distance_change_history)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/prob_change_history.npy',
        prob_change_history)

logger.info(f"Weight history saved: {weight_history.shape}")
logger.info(f"UAM probability history saved: {uam_prob_history.shape}")
logger.info(f"Distance change history saved: {distance_change_history.shape}")
logger.info(f"Probability change history saved: {prob_change_history.shape}")

# Save final vertiport coordinates
# Save as .npy
np.save(os.path.join(centroid_dir, 'optimized_vertiport_coords_final.npy'), vertiport_coords)
# Save as .csv
pd.DataFrame(vertiport_coords, columns=['X', 'Y']).to_csv(
    os.path.join(centroid_dir, 'optimized_vertiport_coords_final.csv'), index=False)

# Save convergence history
pd.DataFrame({'iteration': list(range(1, len(convergence_history) + 1)), 'shift': convergence_history}).to_csv(
    os.path.join(centroid_dir, 'convergence_history.csv'), index=False)

# Save convergence plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(convergence_history) + 1), convergence_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Assignment-based vertiport shift')
plt.title('Vertiport Optimization Convergence')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(centroid_dir, 'convergence_plot.png'))
plt.close()

# Save visualization of centroids and demand points
# Plot initial state
initial_plot_path = os.path.join(centroid_dir, 'initial_centroids_and_demand.png')
plot_centroids_and_demand(
    centroid_history[0],  # Initial centroids
    synthetic_population[['originX', 'originY']].values,
    synthetic_population[['destinationX', 'destinationY']].values,
    0,
    initial_plot_path
)

# Plot final state
final_plot_path = os.path.join(centroid_dir, 'final_centroids_and_demand.png')
plot_centroids_and_demand(
    vertiport_coords,  # Final centroids
    synthetic_population[['originX', 'originY']].values,
    synthetic_population[['destinationX', 'destinationY']].values,
    len(centroid_history) - 1,
    final_plot_path)

# Create animation-like visualization showing centroid evolution (change it to iteration 0 and final)
if len(centroid_history) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Plot key iterations
    key_iterations = [0, len(centroid_history) // 3, 2 * len(centroid_history) // 3, len(centroid_history) - 1]
    titles = ['Initial', '1/3 Progress', '2/3 Progress', 'Final']

    for idx, (iter_idx, title) in enumerate(zip(key_iterations, titles)):
        if iter_idx < len(centroid_history):
            ax = axes[idx]

            # Plot demand points
            ax.scatter(synthetic_population['originX'], synthetic_population['originY'],
                       c='lightblue', s=0.2, alpha=0.4, marker='o')
            ax.scatter(synthetic_population['destinationX'], synthetic_population['destinationY'],
                       c='lightgreen', s=0.2, alpha=0.4, marker='s')

            # Plot centroids
            centroids = centroid_history[iter_idx]
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, alpha=0.8,
                       marker='^', edgecolors='black', linewidth=1)

            ax.set_xlabel('X Coordinate (meters)')
            ax.set_ylabel('Y Coordinate (meters)')
            ax.set_title(f'{title} - Iteration {iter_idx}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    evolution_path = os.path.join(centroid_dir, 'centroid_evolution.png')
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization plots saved to {centroid_dir}")
    logger.info(f"  - Initial state: initial_centroids_and_demand.png")
    logger.info(f"  - Final state: final_centroids_and_demand.png")
    logger.info(f"  - Evolution: centroid_evolution.png")

# 4. FINAL PREDICTION AND OUTPUT

logger.info("Step 4: Final prediction with optimized vertiports and saving results...")

# Recalculate UAM features and probabilities with final optimized vertiports
synthetic_population_with_uam_final = calculate_uam_time_cost(synthetic_population, vertiport_coords, average_car_speed,
                                                              cost_per_m_car)

# For prediction, use only model features:
synthetic_population_with_uam_features = synthetic_population_with_uam_final.copy()
for col in feature_cols:  # Feature Alignment Safeguards
    if col not in synthetic_population_with_uam_features.columns:
        synthetic_population_with_uam_features[col] = 0.0
synthetic_population_with_uam_features = synthetic_population_with_uam_features[feature_cols]

# Get final probabilities
final_proba = predict_mode_probabilities(synthetic_population_with_uam_features, final_model, feature_cols)

# Create final output
output = synthetic_population.copy()
for i, cls in enumerate(classes):
    class_name = class_names.get(cls, f"Class_{cls}")
    output[f'prob_mode_{class_name}'] = final_proba[:, i]

# Add UAM columns from the final calculation
for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel_time_Uam', 'travel_cost_Uam', 'uam_first_mile',
            'uam_last_mile', 'uam_air']:
    output[col] = synthetic_population_with_uam_final[col]

# Save main prediction file with error handling
try:
    os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering', exist_ok=True)
    output_path = '../../../Result/Vertiport_analysis/Probability_clustering/Xgboost_synthetic_population_predictions_weights.csv'
    output.to_csv(output_path, index=False)
    logger.info(f"Main prediction file saved: {output_path}")
except Exception as e:
    logger.error(f"Error saving main prediction file: {e}")

# Save optimized coordinates with error handling
try:
    np.save(
        '../../../Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_weights.npy',
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
        writer.writerow(
            ['Method', 'Converged', 'Iterations', 'Final_Shift', 'Mean_UAM_Probability', 'Std_UAM_Probability',
             'Min_UAM_Probability', 'Max_UAM_Probability'])
        writer.writerow([
            'weights', converged, iterations, final_shift, mean_uam_prob, std_uam_prob, min_uam_prob,
            max_uam_prob
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
        f.write(f"Method: weights (using UAM probabilities with stability controls)\n")
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
logger.info("Results saved with weights and stability controls")

print(f"Iterations: {len(centroid_history) - 1}")
print(f"Final shift: {convergence_history[-1] if convergence_history else 'N/A'}") 