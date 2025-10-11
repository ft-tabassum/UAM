import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import random
import joblib
import os
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# imputer that preserves feature names (required for loading pickle file)
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
    "D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/DataPreprocessing_ML.csv",
    low_memory=False)
# Sample 10% of population
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
# Price testing scenarios:
# Scenario 1: base fare €0 + €3/km
# Scenario 2: base fare €5 + €5/km
uam_cost_m = 0.005  # unit: €/m,     5 €/km (Scenario 2)
uam_base_fare = 5.0  # unit: €,      5 € base fare per trip (Scenario 2)
uam_passenger_capacity = 4  # passengers per UAM vehicle (vehicle specification)
pre_flight_time = 15  # unit : min      (Rothfeld, 2021)
average_car_speed = 418.33  # unit: m/min,    25.1 km/h (TomTom- munich: https://www.tomtom.com/traffic-index/munich-traffic/)
cost_per_m_car = 0.00065  # unit:€/m,       0.65 €/km (Manuscript Number: JTRP-D-24-00632R1)
circuity_factor = 1.215  # for car         (Kim et al., 2025)
# Multimodal catchment areas
car_catchment_distance = 5000  # unit: m,     5 km catchment area for car access
walking_catchment_distance = 1000  # unit: m,  1 km catchment area for walking access
walking_speed = 83.33  # unit: m/min,       5 km/h walking speed (Kim et al., 2025)

# Centroid history tracking
centroid_history = [vertiport_coords.copy()]
convergence_history = []  # Track centroid shifts per iteration
weight_history = []  # Track weights per iteration
uam_prob_history = []  # Track UAM probabilities per iteration
distance_change_history = []  # Track distance matrix changes per iteration


# function to calculate UAM time and travel cost with multimodal access (car + walking)
def calculate_uam_time_cost(df, vertiport_coords, car_speed=average_car_speed, car_cost=cost_per_m_car,
                            uam_speed=uam_cruise_speed, cost_uam_m=uam_cost_m, base_fare=uam_base_fare,
                            pre_flight_time=pre_flight_time, car_catchment=car_catchment_distance,
                            walking_catchment=walking_catchment_distance, walking_speed_param=walking_speed):
    from scipy.spatial.distance import cdist

    # origin(x,y) and destination (x,y) are in meter
    origins = df[['originX', 'originY']].values
    dests = df[['destinationX', 'destinationY']].values

    # Calculate distances to all vertiports
    origin_distances = cdist(origins, vertiport_coords)
    dest_distances = cdist(dests, vertiport_coords)

    # Find nearest vertiport
    origin_v_idx = np.argmin(origin_distances, axis=1)
    dest_v_idx = np.argmin(dest_distances, axis=1)

    # Get distances to nearest vertiports
    origin_dist_to_vertiport = origin_distances[np.arange(len(origins)), origin_v_idx]
    dest_dist_to_vertiport = dest_distances[np.arange(len(dests)), dest_v_idx]

    # Determine access modes based on distance (walking preferred for short distances)
    origin_access_mode = np.where(origin_dist_to_vertiport <= walking_catchment, 'walk', 'car')
    dest_access_mode = np.where(dest_dist_to_vertiport <= walking_catchment, 'walk', 'car')

    # Check catchment area coverage (car catchment is the maximum)
    origin_in_catchment = origin_dist_to_vertiport <= car_catchment
    dest_in_catchment = dest_dist_to_vertiport <= car_catchment

    # Get vertiport coordinates
    origin_v = vertiport_coords[origin_v_idx]
    dest_v = vertiport_coords[dest_v_idx]

    # Calculate access distances based on mode
    first_mile_dist = np.where(
        origin_access_mode == 'walk',
        origin_dist_to_vertiport,  # direct distance for walking
        origin_dist_to_vertiport * circuity_factor  # circuity factor for car
    )

    last_mile_dist = np.where(
        dest_access_mode == 'walk',
        dest_dist_to_vertiport,  # direct distance for walking
        dest_dist_to_vertiport * circuity_factor  # circuity factor for car
    )

    # UAM distance (Euclidean distance in m)
    uam_dist = np.linalg.norm(origin_v - dest_v, axis=1)

    # Calculate access times based on mode in minutes
    first_mile_time = np.where(
        origin_access_mode == 'walk',
        first_mile_dist / walking_speed_param,  # walking speed
        first_mile_dist / car_speed  # car speed
    )

    last_mile_time = np.where(
        dest_access_mode == 'walk',
        last_mile_dist / walking_speed_param,  # walking speed
        last_mile_dist / car_speed  # car speed
    )

    # airborne and total time calculation in min
    airborne_time = uam_dist / uam_speed  # unit: min
    total_time = pre_flight_time + first_mile_time + airborne_time + last_mile_time  # unit: min

    # Calculate access costs based on mode (walking is free)
    first_mile_cost = np.where(
        origin_access_mode == 'walk',
        0,  # walking is free
        first_mile_dist * car_cost  # car cost
    )

    last_mile_cost = np.where(
        dest_access_mode == 'walk',
        0,  # walking is free
        last_mile_dist * car_cost  # car cost
    )
    # Calculate UAM cost: €5.00/km + €5.00 base fare per trip (individual trip cost)
    # Fixed cost per kilometer plus base fare regardless of passenger count
    # Vehicle capacity (4 passengers) is just vehicle specification
    uam_travel_cost = base_fare + (cost_uam_m * uam_dist) + first_mile_cost + last_mile_cost

    df = df.copy()

    # model features (for prediction)
    df['AFT_TT'] = total_time  # min - UAM travel time
    df['AFT_CO'] = uam_travel_cost  # € - UAM travel cost
    # output features (for results)
    df['travel_time_Uam'] = total_time  # min
    df['travel_cost_Uam'] = uam_travel_cost
    df['uam_first_mile'] = first_mile_dist  # m
    df['uam_last_mile'] = last_mile_dist  # m
    df['uam_air'] = uam_dist  # m
    df['uam_origin_vertiport'] = origin_v_idx  # vertiport index
    df['uam_dest_vertiport'] = dest_v_idx  # vertiport index
    # multimodal access information
    df['origin_in_catchment'] = origin_in_catchment  # boolean: origin within car catchment of assigned vertiport
    df['dest_in_catchment'] = dest_in_catchment  # boolean: destination within car catchment of assigned vertiport
    df['origin_to_vertiport_dist'] = origin_dist_to_vertiport  # actual distance to origin vertiport
    df['dest_to_vertiport_dist'] = dest_dist_to_vertiport  # actual distance to destination vertiport
    df['origin_access_mode'] = origin_access_mode  # access mode: 'walk' or 'car'
    df['dest_access_mode'] = dest_access_mode  # access mode: 'walk' or 'car'
    df['origin_access_time'] = first_mile_time  # access time in minutes
    df['dest_access_time'] = last_mile_time  # access time in minutes
    df['origin_access_cost'] = first_mile_cost  # access cost in euros
    df['dest_access_cost'] = last_mile_cost  # access cost in euros
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


# convergence
max_iter = 50
distance_stability_threshold = 0.987  # 9% relative change threshold
converged = False
prev_coords = None
feature_cols = feature_names

# Initialize min_total_shift for convergence tracking
min_total_shift = float('nan')

# Define output directory for price testing - Scenario 2
output_dir = 'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/pricetesting_weighting_clustering/scenario2_base5_perkm5'
os.makedirs(output_dir, exist_ok=True)


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


# Weighted K-Means Implementation (pure weighted mean, no damping)
def weighted_kmeans(X, w, K, max_iter=1000, tol=1e-1,
                    random_state=None):  # Parameters- X : Data points; random_state : Random seed

    rng = np.random.default_rng(random_state)
    eps = 1e-12

    # Log-scale transformation to normalize extreme weight distribution
    w = np.log(1 + w * 100)  # Transform to log scale (multiply by 100 to amplify small values)
    w = w / w.max()  # Normalize to [0,1] range

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
                new_center = mu_w  # Pure weighted mean without damping
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

        # Save intermediate visualization for first iteration (DISABLED - not needed)
        # intermediate_plot_path = os.path.join(output_dir,
        #                                       f'5km_radius_centroids_iteration_{iteration + 1}_unweighted.png')
        # plot_centroids_and_demand(
        #     new_coords,
        #     origins,
        #     dests,
        #     iteration + 1,
        #     intermediate_plot_path
        # )
        # logger.info(
        #     f"First iteration (unweighted) visualization saved: 5km_radius_centroids_iteration_{iteration + 1}_unweighted.png")

    else:
        # SUBSEQUENT ITERATIONS: Weighted K-means based on UAM probabilities
        logger.info(f"Iteration {iteration + 1}: Using weighted K-means based on UAM probabilities")

        # Calculate UAM travel time and cost for each trip
        synthetic_population_with_uam = calculate_uam_time_cost(synthetic_population, vertiport_coords,
                                                                average_car_speed,
                                                                cost_per_m_car,
                                                                base_fare=uam_base_fare)  # synthetic_population_with_uam is the DataFrame with UAM calculations, only ML features, no UAM calculations
        synthetic_population_with_uam_full = synthetic_population_with_uam.copy()  # Keep full version for final output

        # Log multimodal catchment area statistics
        origin_outside = np.sum(~synthetic_population_with_uam['origin_in_catchment'])
        dest_outside = np.sum(~synthetic_population_with_uam['dest_in_catchment'])
        total_trips = len(synthetic_population_with_uam)

        # Count access modes
        origin_walking = np.sum(synthetic_population_with_uam['origin_access_mode'] == 'walk')
        origin_car = np.sum(synthetic_population_with_uam['origin_access_mode'] == 'car')
        dest_walking = np.sum(synthetic_population_with_uam['dest_access_mode'] == 'walk')
        dest_car = np.sum(synthetic_population_with_uam['dest_access_mode'] == 'car')

        logger.info(f"Multimodal catchment area statistics:")
        logger.info(
            f"  Walking catchment: {walking_catchment_distance / 1000:.1f}km, Car catchment: {car_catchment_distance / 1000:.1f}km")
        logger.info(
            f"  Origins outside catchment: {origin_outside}/{total_trips} ({origin_outside / total_trips * 100:.1f}%)")
        logger.info(
            f"  Destinations outside catchment: {dest_outside}/{total_trips} ({dest_outside / total_trips * 100:.1f}%)")
        logger.info(
            f"  Origin access modes - Walking: {origin_walking} ({origin_walking / total_trips * 100:.1f}%), Car: {origin_car} ({origin_car / total_trips * 100:.1f}%)")
        logger.info(
            f"  Destination access modes - Walking: {dest_walking} ({dest_walking / total_trips * 100:.1f}%), Car: {dest_car} ({dest_car / total_trips * 100:.1f}%)")
        logger.info(f"  Max origin distance: {synthetic_population_with_uam['origin_to_vertiport_dist'].max():.0f}m")
        logger.info(f"  Max destination distance: {synthetic_population_with_uam['dest_to_vertiport_dist'].max():.0f}m")

        # Predict mode probabilities
        for col in feature_cols:  # Feature Alignment Safeguards
            if col not in synthetic_population_with_uam.columns:
                synthetic_population_with_uam[col] = 0.0
        synthetic_population_with_uam = synthetic_population_with_uam[feature_cols]
        proba = predict_mode_probabilities(synthetic_population_with_uam, final_model, feature_cols)

        # Use UAM probability as weights for weighted k-means
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

        # Use raw UAM probabilities as weights :log-scale transformation is applied inside weighted_kmeans
        weights = np.concatenate([uam_probs, uam_probs])

        # Store current UAM probabilities for next iteration
        prev_uam_probs = uam_probs.copy()

        # Track weights and probabilities for this iteration (will be updated after feedback loop)
        weight_history.append(weights.copy())
        uam_prob_history.append(uam_probs.copy())

        # Define parameters for weighted k-means clustering
        tol = 1e-1  # Convergence tolerance in meters

        # weighted k-means clustering calculation
        labels, new_coords, kmeans_history = weighted_kmeans(
            X=od_points_current,
            w=weights,
            K=vertiport_k,
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
                                                                        average_car_speed, cost_per_m_car,
                                                                        base_fare=uam_base_fare)

        # Get updated probabilities with new UAM features
        for col in feature_cols:
            if col not in synthetic_population_with_uam_updated.columns:
                synthetic_population_with_uam_updated[col] = 0.0
        synthetic_population_with_uam_updated = synthetic_population_with_uam_updated[feature_cols]
        proba_updated = predict_mode_probabilities(synthetic_population_with_uam_updated, final_model, feature_cols)


        # Apply temperature scaling to reduce model overconfidence
        def temperature_scaling(probabilities, temperature=2.0):
            """Apply temperature scaling to reduce overconfidence"""
            # Convert probabilities to logits
            logits = np.log(probabilities + 1e-8)

            # Scale by temperature
            scaled_logits = logits / temperature

            # Convert back to probabilities and renormalize
            # (model.predict_proba already normalizes, but we need to renormalize after temperature scaling)
            scaled_probs = np.exp(scaled_logits)
            scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)

            return scaled_probs


        # Apply temperature scaling to reduce overconfidence
        proba_updated = temperature_scaling(proba_updated, temperature=2.0)

        # Get updated UAM probabilities (after temperature scaling)
        uam_probs_updated = proba_updated[:, uam_class_idx]

        # ===== COMPREHENSIVE DIAGNOSTIC CHECKS =====
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC: Checking Model Predictions Reasonableness")
        logger.info("=" * 60)

        # 1. Model prediction statistics
        logger.info(f"Model prediction stats:")
        logger.info(f"  Min probability: {proba_updated.min():.6f}")
        logger.info(f"  Max probability: {proba_updated.max():.6f}")
        logger.info(f"  Mean probability: {proba_updated.mean():.6f}")
        logger.info(
            f"  UAM class probabilities - Min: {uam_probs_updated.min():.6f}, Max: {uam_probs_updated.max():.6f}, Mean: {uam_probs_updated.mean():.6f}")

        # 2. Check for extreme values
        if uam_probs_updated.max() > 1.0 or uam_probs_updated.min() < 0.0:
            logger.warning(f"WARNING: UAM probabilities outside [0,1] range!")
            logger.warning(f"  Min: {uam_probs_updated.min():.6f}, Max: {uam_probs_updated.max():.6f}")

        # 3. Check for NaN or infinite values
        if np.isnan(uam_probs_updated).any():
            logger.warning("WARNING: NaN values in UAM probabilities!")
        if np.isinf(uam_probs_updated).any():
            logger.warning("WARNING: Infinite values in UAM probabilities!")

        # 4. Compare with previous iteration
        if iteration > 0:  # Not first iteration
            prob_diff = np.abs(uam_probs_updated - uam_probs)
            logger.info(
                f"Probability differences - Min: {prob_diff.min():.6f}, Max: {prob_diff.max():.6f}, Mean: {prob_diff.mean():.6f}")

            # Check if changes are reasonable
            if prob_diff.max() > 0.5:  # More than 50% change
                logger.warning(f"WARNING: Large probability change detected! Max change: {prob_diff.max():.6f}")

        # 5. Check AFT feature ranges
        if 'AFT_TT' in synthetic_population_with_uam_updated.columns:
            aft_tt = synthetic_population_with_uam_updated['AFT_TT']
            logger.info(f"AFT_TT range: {aft_tt.min():.2f} to {aft_tt.max():.2f} minutes")
            if aft_tt.max() > 300:  # More than 5 hours
                logger.warning("WARNING: AFT_TT values seem too high!")
            if aft_tt.min() < 0:
                logger.warning("WARNING: AFT_TT has negative values!")

        if 'AFT_CO' in synthetic_population_with_uam_updated.columns:
            aft_co = synthetic_population_with_uam_updated['AFT_CO']
            logger.info(f"AFT_CO range: {aft_co.min():.2f} to {aft_co.max():.2f} euros")
            if aft_co.max() > 1000:  # More than 1000 euros
                logger.warning("WARNING: AFT_CO values seem too high!")
            if aft_co.min() < 0:
                logger.warning("WARNING: AFT_CO has negative values!")

        # 6. Check for extreme probability distributions
        prob_hist, _ = np.histogram(uam_probs_updated, bins=10, range=(0, 1))
        logger.info(f"UAM probability distribution (10 bins): {prob_hist}")

        # Check if most probabilities are near 0 or 1 (model being too confident)
        near_zero = np.sum(uam_probs_updated < 0.1)
        near_one = np.sum(uam_probs_updated > 0.9)
        total = len(uam_probs_updated)
        logger.info(f"Probabilities near 0 (<0.1): {near_zero}/{total} ({near_zero / total * 100:.1f}%)")
        logger.info(f"Probabilities near 1 (>0.9): {near_one}/{total} ({near_one / total * 100:.1f}%)")

        if near_zero + near_one > total * 0.8:  # More than 80% are extreme
            logger.warning("WARNING: Model is making too many extreme predictions!")

        logger.info("=" * 60)
        logger.info("END DIAGNOSTIC CHECKS")
        logger.info("=" * 60)

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

            # Track convergence metrics
            distance_change_history.append(distance_change)

            logger.info(
                f"Iteration {iteration + 1}: Coordinate shift = {min_total_shift:.2f}m, Distance matrix change = {distance_change:.6f} (stable: {distance_stable})")

            # Save intermediate visualization every 5 iterations (DISABLED - not needed)
            # if (iteration + 1) % 5 == 0 or iteration < 5:
            #     intermediate_plot_path = os.path.join(output_dir,
            #                                           f'5km_radius_centroids_iteration_{iteration + 1}.png')
            #     plot_centroids_and_demand(
            #         new_coords_ordered,
            #         origins,
            #         dests,
            #         iteration + 1,
            #         intermediate_plot_path
            #     )
            #     logger.info(f"Intermediate visualization saved: 5km_radius_centroids_iteration_{iteration + 1}.png")

            # Check for convergence - distance stability
            layout_converged = distance_stable

            # distance matrix stability required for convergence
            if layout_converged and iteration >= 2:
                logger.info(
                    f"Converged after {iteration + 1} iterations with distance matrix stability: {distance_change:.6f}")
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

# Save ONLY weight history and final vertiport coordinates (no centroids)
weight_history = np.array(weight_history)  # shape: (num_iterations, 2*num_trips)
uam_prob_history = np.array(uam_prob_history)  # shape: (num_iterations, num_trips)

# Save weight history
np.save(os.path.join(output_dir, 'weight_history.npy'), weight_history)
logger.info(f"Weight history saved: {weight_history.shape}")

# Save final vertiport coordinates
# Save as .npy
np.save(os.path.join(output_dir, 'optimized_vertiport_coords.npy'), vertiport_coords)
# Save as .csv
pd.DataFrame(vertiport_coords, columns=['X', 'Y']).to_csv(
    os.path.join(output_dir, 'optimized_vertiport_coords.csv'), index=False)
logger.info(f"Final vertiport coordinates saved to {output_dir}")

# 4. FINAL PREDICTION AND OUTPUT

logger.info("Step 4: Final prediction with optimized vertiports and saving results...")

# Recalculate UAM features and probabilities with final optimized vertiports
synthetic_population_with_uam_final = calculate_uam_time_cost(synthetic_population, vertiport_coords, average_car_speed,
                                                              cost_per_m_car, base_fare=uam_base_fare)

# For prediction, use only model features:
synthetic_population_with_uam_features = synthetic_population_with_uam_final.copy()
for col in feature_cols:  # Feature Alignment Safeguards
    if col not in synthetic_population_with_uam_features.columns:
        synthetic_population_with_uam_features[col] = 0.0
synthetic_population_with_uam_features = synthetic_population_with_uam_features[feature_cols]

# Get final probabilities
final_proba = predict_mode_probabilities(synthetic_population_with_uam_features, final_model, feature_cols)


# Apply temperature scaling to final probabilities (consistent with optimization iterations)
def temperature_scaling_final(probabilities, temperature=2.0):
    """Apply temperature scaling to reduce overconfidence"""
    # Convert probabilities to logits
    logits = np.log(probabilities + 1e-8)

    # Scale by temperature
    scaled_logits = logits / temperature

    # Convert back to probabilities and renormalize
    scaled_probs = np.exp(scaled_logits)
    scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)

    return scaled_probs


logger.info("Applying temperature scaling to final probabilities (temperature=2.0)...")
final_proba = temperature_scaling_final(final_proba, temperature=2.0)
logger.info(f"Final UAM probability after scaling - Mean: {final_proba[:, uam_class_idx].mean():.4f}")

# Create final output
output = synthetic_population.copy()
for i, cls in enumerate(classes):
    class_name = class_names.get(cls, f"Class_{cls}")
    output[f'prob_mode_{class_name}'] = final_proba[:, i]

# Add UAM columns from the final calculation
for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel_time_Uam', 'travel_cost_Uam', 'uam_first_mile',
            'uam_last_mile', 'uam_air', 'origin_in_catchment', 'dest_in_catchment',
            'origin_to_vertiport_dist', 'dest_to_vertiport_dist', 'origin_access_mode', 'dest_access_mode',
            'origin_access_time', 'dest_access_time', 'origin_access_cost', 'dest_access_cost']:
    output[col] = synthetic_population_with_uam_final[col]

# Save main prediction file (not needed for this testing)
# Only saving weights and vertiport locations per user request

# =========================
# 5. SAVE SUMMARY AND REPORT
# =========================
import csv

report_path = os.path.join(output_dir, 'method_report.txt')

# Save summary and report with error handling
try:
    # Create directory for summary file (already created)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    mean_uam_prob = float(np.mean(uam_probs))
    std_uam_prob = float(np.std(uam_probs))
    min_uam_prob = float(np.min(uam_probs))
    max_uam_prob = float(np.max(uam_probs))
    final_shift = float(convergence_history[-1]) if convergence_history else float('nan')
    iterations = len(centroid_history) - 1

    # Calculate multimodal catchment area statistics
    origin_outside_final = np.sum(~output['origin_in_catchment'])
    dest_outside_final = np.sum(~output['dest_in_catchment'])
    total_trips_final = len(output)
    max_origin_dist_final = float(output['origin_to_vertiport_dist'].max())
    max_dest_dist_final = float(output['dest_to_vertiport_dist'].max())

    # Count access modes
    origin_walking_final = np.sum(output['origin_access_mode'] == 'walk')
    origin_car_final = np.sum(output['origin_access_mode'] == 'car')
    dest_walking_final = np.sum(output['dest_access_mode'] == 'walk')
    dest_car_final = np.sum(output['dest_access_mode'] == 'car')

    # Save summary CSV
    summary_path = os.path.join(output_dir, 'method_summary.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['Method', 'Converged', 'Iterations', 'Final_Shift', 'Mean_UAM_Probability', 'Std_UAM_Probability',
             'Min_UAM_Probability', 'Max_UAM_Probability', 'Origins_Outside_Catchment', 'Dest_Outside_Catchment',
             'Max_Origin_Dist', 'Max_Dest_Dist', 'Car_Catchment_Distance', 'Walking_Catchment_Distance',
             'Origin_Walking_Count', 'Origin_Car_Count', 'Dest_Walking_Count', 'Dest_Car_Count'])
        writer.writerow([
            f'price_test_base{uam_base_fare}_per_km{uam_cost_m*1000}', converged, iterations, final_shift, mean_uam_prob, std_uam_prob, min_uam_prob,
            max_uam_prob, origin_outside_final, dest_outside_final, max_origin_dist_final, max_dest_dist_final,
            car_catchment_distance, walking_catchment_distance, origin_walking_final, origin_car_final,
            dest_walking_final, dest_car_final
        ])
    logger.info(f"Summary CSV saved: {summary_path}")

    # Save detailed text report
    with open(report_path, 'w') as f:
        f.write(f"VERTIPORT OPTIMIZATION REPORT - SCENARIO 2\n")
        f.write(f"{'=' * 50}\n")
        from datetime import datetime

        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"SUMMARY:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"Method: weights_multimodal (using UAM probabilities with multimodal access)\n")
        f.write(
            f"UAM Pricing Model: €{uam_base_fare:.2f} base fare + €{uam_cost_m * 1000:.2f}/km per trip (individual trip cost)\n")
        f.write(f"UAM Vehicle Capacity: {uam_passenger_capacity} passengers (vehicle specification)\n")
        f.write(f"Converged: {converged}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Final Shift: {final_shift:.6f}\n")
        f.write(f"Mean UAM Probability: {mean_uam_prob:.4f}\n")
        f.write(f"UAM Probability Std: {std_uam_prob:.4f}\n")
        f.write(f"UAM Probability Range: [{min_uam_prob:.4f}, {max_uam_prob:.4f}]\n")
        f.write(f"\nMULTIMODAL CATCHMENT AREA STATISTICS:\n")
        f.write(f"Walking Catchment Distance: {walking_catchment_distance / 1000:.1f} km\n")
        f.write(f"Car Catchment Distance: {car_catchment_distance / 1000:.1f} km\n")
        f.write(
            f"Origins outside catchment: {origin_outside_final}/{total_trips_final} ({origin_outside_final / total_trips_final * 100:.1f}%)\n")
        f.write(
            f"Destinations outside catchment: {dest_outside_final}/{total_trips_final} ({dest_outside_final / total_trips_final * 100:.1f}%)\n")
        f.write(
            f"Origin access modes - Walking: {origin_walking_final} ({origin_walking_final / total_trips_final * 100:.1f}%), Car: {origin_car_final} ({origin_car_final / total_trips_final * 100:.1f}%)\n")
        f.write(
            f"Destination access modes - Walking: {dest_walking_final} ({dest_walking_final / total_trips_final * 100:.1f}%), Car: {dest_car_final} ({dest_car_final / total_trips_final * 100:.1f}%)\n")
        f.write(f"Max origin distance: {max_origin_dist_final:.0f}m ({max_origin_dist_final / 1000:.1f}km)\n")
        f.write(f"Max destination distance: {max_dest_dist_final:.0f}m ({max_dest_dist_final / 1000:.1f}km)\n")
        f.write(f"\nCONVERGENCE HISTORY (first 20 shown):\n")
        for i, shift in enumerate(convergence_history[:20]):
            f.write(f"  Iter {i + 1}: {shift:.6f}\n")
        if len(convergence_history) > 20:
            f.write(f"  ... ({len(convergence_history) - 20} more)\n")
        f.write(f"\nAll results saved to {output_dir}\n")

    logger.info(f"Detailed report saved: {report_path}")
    logger.info("Step 5 complete: Summary and report saved.")

except Exception as e:
    logger.error(f"Error saving summary and report: {e}")
logger.info("Results saved with weights and stability controls")

print(f"Iterations: {len(centroid_history) - 1}")
print(f"Final shift: {convergence_history[-1] if convergence_history else 'N/A'}")

