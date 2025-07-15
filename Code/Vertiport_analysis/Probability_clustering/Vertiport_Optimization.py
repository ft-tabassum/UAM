import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
import random
import os
import pickle

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create output directories
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Centroid', exist_ok=True)

# Load the trained model from Part 1
logger.info("Loading trained XGBoost model from Part 1...")
with open('../../../Result/Vertiport_analysis/Probability_clustering/Trained_Models/xgboost_model_LighterModel.pkl', 'rb') as f:
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
#synthetic_population = pd.read_csv("D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/synthetic_population_processing.csv")
synthetic_population = pd.read_csv("D:/Thesis/UAM/Result/Scenario/destination_Zone.csv") #only for 80933 PLZ
# =========================
# 2. INITIALIZE K-MEANS++ WITH 20 VERTIPORTS
# =========================
logger.info("Step 2: Initializing k-means++ with 20 vertiports on O/D points from synthetic population data...")
od_points = np.vstack([
    synthetic_population[['originX', 'originY']].values,
    synthetic_population[['destinationX', 'destinationY']].values
])
kmeans = KMeans(n_clusters=20, init='k-means++', random_state=RANDOM_SEED)
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

# Set car speed and cost per km to default values
avg_car_speed = 40  # Default car speed in km/h
car_cost_per_km = 0.25  # Default car cost in €/km
logger.info(f"Using car speed: {avg_car_speed:.2f} km/h, car cost per km: {car_cost_per_km:.2f} €/km")

# --- Centroid history tracking ---
centroid_history = [vertiport_coords.copy()]

# UAM calculation function, based on assumptions from the literature
VERTIPORT_K = 20
UAM_CRUISE_SPEED_KMH = 350
UAM_COST_PER_KM = 1.0
BASE_FARE = 18.4
PRE_FLIGHT_TIME_HOURS = 15 / 60


# --- Weight normalization functions ---
def softmax(x, temperature=1.0):
    """Global softmax function (original)"""
    x = np.array(x)
    x = x / temperature
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()

def normalize_weights_within_clusters(weights, cluster_labels, method='simple', temperature=1.0):
    """
    Normalize weights so that each cluster has weight sum = 1
    
    Args:
        weights: array of weights for all points
        cluster_labels: cluster assignment for each point
        method: 'simple', 'softmax', or 'log'
        temperature: temperature for softmax (only used if method='softmax')
    
    Returns:
        normalized_weights: weights normalized within each cluster
    """
    normalized_weights = np.zeros_like(weights)
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_weights = weights[cluster_mask]
        
        if method == 'simple':
            # Simple normalization: x / sum(x)
            if np.sum(cluster_weights) > 0:
                normalized_cluster_weights = cluster_weights / np.sum(cluster_weights)
            else:
                normalized_cluster_weights = np.ones_like(cluster_weights) / len(cluster_weights)
                
        elif method == 'softmax':
            # Softmax within cluster: exp(x/t) / sum(exp(x/t))
            if np.sum(cluster_weights) > 0:
                cluster_weights_scaled = cluster_weights / temperature
                e_x = np.exp(cluster_weights_scaled - np.max(cluster_weights_scaled))
                normalized_cluster_weights = e_x / np.sum(e_x)
            else:
                normalized_cluster_weights = np.ones_like(cluster_weights) / len(cluster_weights)
                
        elif method == 'log':
            # Log transformation: log(1 + x) then normalize
            if np.sum(cluster_weights) > 0:
                log_weights = np.log(1 + cluster_weights)
                normalized_cluster_weights = log_weights / np.sum(log_weights)
            else:
                normalized_cluster_weights = np.ones_like(cluster_weights) / len(cluster_weights)
        
        normalized_weights[cluster_mask] = normalized_cluster_weights
    
    return normalized_weights


def calculate_uam_time_cost(df, vertiport_coords, car_speed, car_cost_km, base_fare=BASE_FARE,
                            uam_speed=UAM_CRUISE_SPEED_KMH, uam_cost_km=UAM_COST_PER_KM,
                            pre_flight_time=PRE_FLIGHT_TIME_HOURS):
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
    uam_dist = np.linalg.norm(origin_v - dest_v, axis=1)
    first_mile_time = first_mile_dist / car_speed
    last_mile_time = last_mile_dist / car_speed
    airborne_time = uam_dist / uam_speed
    total_time = pre_flight_time + first_mile_time + airborne_time + last_mile_time
    first_mile_cost = first_mile_dist * car_cost_km
    last_mile_cost = last_mile_dist * car_cost_km
    uam_cost = base_fare + (uam_cost_km * uam_dist) + first_mile_cost + last_mile_cost
    df = df.copy()
    df['travel time_Uam'] = total_time
    df['TravelCost_Uam'] = uam_cost
    df['uam_first_mile_km'] = first_mile_dist
    df['uam_last_mile_km'] = last_mile_dist
    df['uam_air_km'] = uam_dist
    df['uam_origin_vertiport'] = origin_v_idx
    df['uam_dest_vertiport'] = dest_v_idx
    return df


def predict_mode_probabilities(df, model, feature_cols): #features are arranged identically to how they were during training
    X = df[feature_cols]
    return model.predict_proba(X)


max_iter = 100000
convergence_threshold = 1e-1  # Convergence threshold for vertiport shift
converged = False
prev_coords = None
feature_cols = feature_names

# Choose normalization method: 'simple', 'softmax', or 'log'
NORMALIZATION_METHOD = 'softmax'  # Change this to test different methods
logger.info(f"Using {NORMALIZATION_METHOD} normalization within clusters")

for iteration in range(max_iter):
    logger.info(f"Iteration {iteration + 1}...")
    # a. Calculate UAM travel time and cost for each trip
    synthetic_population_with_uam = calculate_uam_time_cost(synthetic_population, vertiport_coords, avg_car_speed, car_cost_per_km)
    # b. Add these UAM features to the synthetic population data (already done in synthetic_population_with_uam)
    # c. Predict mode probabilities
    for col in feature_cols: #Feature Alignment Safeguards
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
    
    # e. Apply normalization within clusters after each clustering iteration
    # First, get current cluster assignments using previous centroids (or initial centroids for first iteration)
    from scipy.spatial.distance import cdist
    origins = synthetic_population[['originX', 'originY']].values
    dests = synthetic_population[['destinationX', 'destinationY']].values
    od_points_current = np.vstack([origins, dests])
    
    # Assign points to nearest centroids
    distances = cdist(od_points_current, vertiport_coords)
    cluster_labels = np.argmin(distances, axis=1)
    
    # Normalize weights within each cluster
    normalized_weights = normalize_weights_within_clusters(
        uam_probs, cluster_labels[:len(uam_probs)], 
        method=NORMALIZATION_METHOD, temperature=1.0
    )
    weights = np.concatenate([normalized_weights, normalized_weights])
    
    # f. Perform weighted k-means clustering
    kmeans = KMeans(n_clusters=VERTIPORT_K, init='k-means++', random_state=RANDOM_SEED)
    kmeans.fit(od_points, sample_weight=weights)
    new_coords = kmeans.cluster_centers_
    
    # g. Check convergence
    if prev_coords is not None:
        shift = np.linalg.norm(new_coords - prev_coords)
        logger.info(f"Vertiport shift: {shift:.6f}")
        if shift < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations with shift: {shift:.6f}")
            converged = True
            centroid_history.append(new_coords.copy())
            break
    prev_coords = new_coords
    vertiport_coords = new_coords
    centroid_history.append(vertiport_coords.copy())

if not converged:
    logger.warning(f"Did not converge within {max_iter} iterations. Final shift: {shift:.6f}")

logger.info("Step 3 complete: Vertiport optimization finished.")

# Save centroid history after optimization
centroid_history = np.array(centroid_history)  # shape: (num_iterations+1, 20, 2)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/vertiport_centroid_history.npy', centroid_history)

# =========================
# 4. FINAL PREDICTION AND OUTPUT
# =========================
logger.info("Step 4: Final prediction with optimized vertiports and saving results...")
# Calculate UAM features (full DataFrame)
synthetic_population_with_uam_full = calculate_uam_time_cost(synthetic_population, vertiport_coords, avg_car_speed, car_cost_per_km)
# For prediction, use only model features:
synthetic_population_with_uam = synthetic_population_with_uam_full.copy()
for col in feature_cols: #Feature Alignment Safeguards
    if col not in synthetic_population_with_uam.columns:
        synthetic_population_with_uam[col] = 0.0
synthetic_population_with_uam = synthetic_population_with_uam[feature_cols]
proba = predict_mode_probabilities(synthetic_population_with_uam, final_model, feature_cols)
output = synthetic_population.copy()
for i, cls in enumerate(classes):
    class_name = class_names.get(cls, f"Class_{cls}")
    output[f'prob_mode_{class_name}'] = proba[:, i]
# Add UAM columns from the full DataFrame
for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel time_Uam', 'TravelCost_Uam', 'uam_first_mile_km',
            'uam_last_mile_km', 'uam_air_km']:
    output[col] = synthetic_population_with_uam_full[col]
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering', exist_ok=True)
output.to_csv(
    f'../../../Result/Vertiport_analysis/Probability_clustering/Xgboost_synthetic_population_predictions_{NORMALIZATION_METHOD}.csv',
    index=False)
np.save(f'../../../Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_{NORMALIZATION_METHOD}.npy', vertiport_coords)
logger.info("Step 4 complete: All results saved to Result/Vertiport_analysis/Probability_clustering/")
logger.info(f"Results saved with {NORMALIZATION_METHOD} normalization method") 