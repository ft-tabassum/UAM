import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
import random
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create output directories
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results', exist_ok=True)

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
synthetic_population = pd.read_csv("D:/Thesis/UAM/Result/Scenario/moosach_related_trips.csv") #only for 80933 PLZ

# =========================
# CONSTANTS AND FUNCTIONS
# =========================
VERTIPORT_K = 20
UAM_CRUISE_SPEED_KMH = 350
UAM_COST_PER_KM = 1.0
BASE_FARE = 18.4
PRE_FLIGHT_TIME_HOURS = 15 / 60

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

def predict_mode_probabilities(df, model, feature_cols):
    X = df[feature_cols]
    return model.predict_proba(X)

def run_optimization_with_method(method_name, synthetic_population, final_model, feature_names, classes, class_names):
    """
    Run the vertiport optimization with a specific normalization method
    
    Returns:
        dict: Results including final coordinates, convergence info, and predictions
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Running optimization with {method_name} normalization")
    logger.info(f"{'='*50}")
    
    # Initialize k-means++ with 20 vertiports
    od_points = np.vstack([
        synthetic_population[['originX', 'originY']].values,
        synthetic_population[['destinationX', 'destinationY']].values
    ])
    kmeans = KMeans(n_clusters=20, init='k-means++', random_state=RANDOM_SEED)
    kmeans.fit(od_points)
    vertiport_coords = kmeans.cluster_centers_
    
    # Set parameters
    max_iter = 3000
    convergence_threshold = 1e-1
    converged = False
    prev_coords = None
    feature_cols = feature_names
    avg_car_speed = 40
    car_cost_per_km = 0.25
    
    # Track convergence history
    convergence_history = []
    centroid_history = [vertiport_coords.copy()]
    
    for iteration in range(max_iter):
        if iteration % 100 == 0:
            logger.info(f"Iteration {iteration + 1}...")
            
        # a. Calculate UAM travel time and cost for each trip
        synthetic_population_with_uam = calculate_uam_time_cost(synthetic_population, vertiport_coords, avg_car_speed, car_cost_per_km)
        
        # b. Predict mode probabilities
        for col in feature_cols:
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
        
        uam_probs = proba[:, uam_class_idx]
        
        # d. Apply normalization within clusters
        from scipy.spatial.distance import cdist
        origins = synthetic_population[['originX', 'originY']].values
        dests = synthetic_population[['destinationX', 'destinationY']].values
        od_points_current = np.vstack([origins, dests])
        
        distances = cdist(od_points_current, vertiport_coords)
        cluster_labels = np.argmin(distances, axis=1)
        
        normalized_weights = normalize_weights_within_clusters(
            uam_probs, cluster_labels[:len(uam_probs)], 
            method=method_name, temperature=1.0
        )
        weights = np.concatenate([normalized_weights, normalized_weights])
        
        # e. Perform weighted k-means clustering
        kmeans = KMeans(n_clusters=VERTIPORT_K, init='k-means++', random_state=RANDOM_SEED)
        kmeans.fit(od_points, sample_weight=weights)
        new_coords = kmeans.cluster_centers_
        
        # f. Check convergence
        if prev_coords is not None:
            shift = np.linalg.norm(new_coords - prev_coords)
            convergence_history.append(shift)
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
    
    # Final prediction
    synthetic_population_with_uam_full = calculate_uam_time_cost(synthetic_population, vertiport_coords, avg_car_speed, car_cost_per_km)
    synthetic_population_with_uam = synthetic_population_with_uam_full.copy()
    for col in feature_cols:
        if col not in synthetic_population_with_uam.columns:
            synthetic_population_with_uam[col] = 0.0
    synthetic_population_with_uam = synthetic_population_with_uam[feature_cols]
    proba = predict_mode_probabilities(synthetic_population_with_uam, final_model, feature_cols)
    
    output = synthetic_population.copy()
    for i, cls in enumerate(classes):
        class_name = class_names.get(cls, f"Class_{cls}")
        output[f'prob_mode_{class_name}'] = proba[:, i]
    
    for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel time_Uam', 'TravelCost_Uam', 'uam_first_mile_km',
                'uam_last_mile_km', 'uam_air_km']:
        output[col] = synthetic_population_with_uam_full[col]
    
    return {
        'method': method_name,
        'final_coords': vertiport_coords,
        'converged': converged,
        'iterations': len(centroid_history) - 1,
        'final_shift': convergence_history[-1] if convergence_history else None,
        'convergence_history': convergence_history,
        'centroid_history': np.array(centroid_history),
        'predictions': output,
        'uam_probabilities': proba[:, uam_class_idx]
    }

# =========================
# RUN COMPARISON
# =========================
logger.info("Starting comparison of three weight transformation methods...")

methods = ['simple', 'softmax', 'log']
results = {}

for method in methods:
    try:
        result = run_optimization_with_method(method, synthetic_population, final_model, feature_names, classes, class_names)
        results[method] = result
        logger.info(f"Completed {method} method successfully")
    except Exception as e:
        logger.error(f"Error in {method} method: {str(e)}")
        results[method] = None

# =========================
# COMPARISON ANALYSIS
# =========================
logger.info("\n" + "="*60)
logger.info("COMPARISON RESULTS")
logger.info("="*60)

comparison_data = []
for method, result in results.items():
    if result is not None:
        comparison_data.append({
            'Method': method,
            'Converged': result['converged'],
            'Iterations': result['iterations'],
            'Final_Shift': result['final_shift'],
            'Mean_UAM_Probability': np.mean(result['uam_probabilities']),
            'Std_UAM_Probability': np.std(result['uam_probabilities']),
            'Min_UAM_Probability': np.min(result['uam_probabilities']),
            'Max_UAM_Probability': np.max(result['uam_probabilities'])
        })
        logger.info(f"\n{method.upper()} METHOD:")
        logger.info(f"  Converged: {result['converged']}")
        logger.info(f"  Iterations: {result['iterations']}")
        logger.info(f"  Final Shift: {result['final_shift']:.6f}")
        logger.info(f"  Mean UAM Probability: {np.mean(result['uam_probabilities']):.4f}")
        logger.info(f"  UAM Probability Range: [{np.min(result['uam_probabilities']):.4f}, {np.max(result['uam_probabilities']):.4f}]")

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/method_comparison_summary.csv', index=False)

# =========================
# VISUALIZATION
# =========================
logger.info("\nCreating comparison visualizations...")

# Create output directory for plots
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/Plots', exist_ok=True)

# 1. Convergence comparison
plt.figure(figsize=(15, 10))

# Convergence curves
plt.subplot(2, 3, 1)
for method, result in results.items():
    if result is not None and result['convergence_history']:
        plt.plot(result['convergence_history'], label=method, alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('Vertiport Shift')
plt.title('Convergence Comparison')
plt.legend()
plt.yscale('log')

# 2. UAM probability distributions
plt.subplot(2, 3, 2)
for method, result in results.items():
    if result is not None:
        plt.hist(result['uam_probabilities'], bins=50, alpha=0.6, label=method, density=True)
plt.xlabel('UAM Probability')
plt.ylabel('Density')
plt.title('UAM Probability Distribution')
plt.legend()

# 3. Final vertiport locations comparison
plt.subplot(2, 3, 3)
colors = ['red', 'blue', 'green']
for i, (method, result) in enumerate(results.items()):
    if result is not None:
        coords = result['final_coords']
        plt.scatter(coords[:, 0], coords[:, 1], c=colors[i], label=method, alpha=0.7, s=50)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Final Vertiport Locations')
plt.legend()

# 4. Box plot of UAM probabilities
plt.subplot(2, 3, 4)
prob_data = []
prob_labels = []
for method, result in results.items():
    if result is not None:
        prob_data.append(result['uam_probabilities'])
        prob_labels.extend([method] * len(result['uam_probabilities']))
if prob_data:
    plt.boxplot(prob_data, labels=list(results.keys()))
plt.ylabel('UAM Probability')
plt.title('UAM Probability Distribution (Box Plot)')

# 5. Iterations comparison
plt.subplot(2, 3, 5)
methods_list = []
iterations_list = []
for method, result in results.items():
    if result is not None:
        methods_list.append(method)
        iterations_list.append(result['iterations'])
plt.bar(methods_list, iterations_list)
plt.ylabel('Number of Iterations')
plt.title('Convergence Speed Comparison')

# 6. Final shift comparison
plt.subplot(2, 3, 6)
shifts = []
for method, result in results.items():
    if result is not None and result['final_shift'] is not None:
        shifts.append(result['final_shift'])
    else:
        shifts.append(0)
plt.bar(list(results.keys()), shifts)
plt.ylabel('Final Shift')
plt.title('Final Convergence Quality')

plt.tight_layout()
plt.savefig('../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/Plots/comparison_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================
# SAVE RESULTS
# =========================
logger.info("Saving detailed results...")

# Save individual results
for method, result in results.items():
    if result is not None:
        # Save predictions
        result['predictions'].to_csv(
            f'../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/predictions_{method}.csv',
            index=False
        )
        
        # Save final coordinates
        np.save(
            f'../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/final_coords_{method}.npy',
            result['final_coords']
        )
        
        # Save centroid history
        np.save(
            f'../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/centroid_history_{method}.npy',
            result['centroid_history']
        )

# Save comparison summary
comparison_df.to_csv('../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/method_comparison_summary.csv', index=False)

# Create detailed comparison report
with open('../../../Result/Vertiport_analysis/Probability_clustering/Comparison_Results/comparison_report.txt', 'w') as f:
    f.write("VERTIPORT OPTIMIZATION METHOD COMPARISON REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("SUMMARY:\n")
    f.write("-" * 20 + "\n")
    for _, row in comparison_df.iterrows():
        f.write(f"\n{row['Method'].upper()} METHOD:\n")
        f.write(f"  Converged: {row['Converged']}\n")
        f.write(f"  Iterations: {row['Iterations']}\n")
        f.write(f"  Final Shift: {row['Final_Shift']:.6f}\n")
        f.write(f"  Mean UAM Probability: {row['Mean_UAM_Probability']:.4f}\n")
        f.write(f"  UAM Probability Std: {row['Std_UAM_Probability']:.4f}\n")
        f.write(f"  UAM Probability Range: [{row['Min_UAM_Probability']:.4f}, {row['Max_UAM_Probability']:.4f}]\n")
    
    f.write("\n\nRECOMMENDATIONS:\n")
    f.write("-" * 20 + "\n")
    
    # Find best method for each metric
    best_convergence = comparison_df.loc[comparison_df['Final_Shift'].idxmin(), 'Method']
    best_speed = comparison_df.loc[comparison_df['Iterations'].idxmin(), 'Method']
    highest_uam_prob = comparison_df.loc[comparison_df['Mean_UAM_Probability'].idxmax(), 'Method']
    
    f.write(f"Best convergence quality: {best_convergence}\n")
    f.write(f"Fastest convergence: {best_speed}\n")
    f.write(f"Highest mean UAM probability: {highest_uam_prob}\n")

logger.info("Comparison complete! Results saved to:")
logger.info("  - Comparison_Results/method_comparison_summary.csv")
logger.info("  - Comparison_Results/comparison_report.txt")
logger.info("  - Comparison_Results/Plots/comparison_analysis.png")
logger.info("  - Individual results for each method in Comparison_Results/") 