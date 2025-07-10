import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import Counter
import os
from sklearn.cluster import KMeans
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create output directories
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Testing_Probabilities', exist_ok=True)
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Training_Probabilities', exist_ok=True)
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Feature_Importance', exist_ok=True)
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Confusion_Matrix', exist_ok=True)
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering/Prediction_EvaluationMetrics', exist_ok=True)

# Load LighterModel data (UAM-aware data for training)
## Normalization is skipped because it generates many zero values, which leads to incorrect results.

logger.info("Loading processed LighterModel data...")
lighter_data = pd.read_csv("D:/Thesis/UAM/Result/Vertiport_analysis/LighterModel/LighterModel_processing.csv")

# Load trial data (UAM-unaware data for prediction)
logger.info("Loading processed trial data...")
trial_data = pd.read_csv("D:/Thesis/UAM/Result/Vertiport_analysis/Synthetic_population/synthetic_population_processing.csv")

# Define features and target for LighterModel
y_lighter = lighter_data['tmode']
X_lighter = lighter_data.drop(columns=['tmode'])

# Original classes
classes = np.unique(y_lighter)
n_classes = len(classes)
logger.info(f"Number of classes: {n_classes}")
logger.info(f"Classes: {classes}")

# Create base pipeline
base_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('classifier', XGBClassifier(random_state=RANDOM_SEED))
])

# Hyperparameter grid - focused on key parameters
param_grid = {
    'classifier__n_estimators': [90, 100, 110],
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.001, 0.005, 0.01],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 0.9, 1.0]
}

# Split LighterModel data into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_lighter, y_lighter, test_size=0.2, random_state=RANDOM_SEED, stratify=y_lighter
)

# Setup cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Store feature names
feature_names = X_lighter.columns.tolist()

# Initialize storage for metrics
fold_metrics = {
    'accuracies': [], 'precisions': [], 'recalls': [],
    'f1s': [], 'roc_aucs': [], 'confusion_matrices': [],
    'probabilities': [], 'true_labels': [], 'pred_labels': [],
    'best_params': [], 'train_accuracies': [], 'class_accuracies': [],
    'feature_importances': []
}

# Initialize a list to store the probabilities
all_fold_probs = []

# Perform cross-validation with GridSearchCV in each fold
logger.info("Step 1: Training, validating, and testing XGBoost model on LighterModel data...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val), 1):
    logger.info(f"CV Fold {fold}/10...")

    # Split data for this fold
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    # Use GridSearchCV for hyperparameter tuning on training data
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=5,  # Use 5-fold CV for hyperparameter tuning
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit GridSearchCV on training data
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Calculate training accuracy
    train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    fold_metrics['train_accuracies'].append(train_acc)

    # Make predictions on validation set
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)

    # Calculate per-class accuracy
    class_acc = {}
    for cls in classes:
        mask = y_val == cls
        if np.any(mask):
            class_acc[cls] = accuracy_score(y_val[mask], val_pred[mask])
        else:
            class_acc[cls] = np.nan
    fold_metrics['class_accuracies'].append(class_acc)

    # Append the probabilities to the list (add fold number as a column)
    fold_probs_df = pd.DataFrame(val_proba, columns=classes)
    fold_probs_df['fold'] = fold  # Add fold number to distinguish rows
    all_fold_probs.append(fold_probs_df)

    # Store feature importances
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    fold_metrics['feature_importances'].append(feature_importances)

    # Calculate metrics
    val_acc = accuracy_score(y_val, val_pred)
    val_prec = precision_score(y_val, val_pred, average='weighted', zero_division=0)
    val_rec = recall_score(y_val, val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)
    val_cm = confusion_matrix(y_val, val_pred, labels=classes)

    # Calculate ROC AUC
    y_val_bin = label_binarize(y_val, classes=classes)
    try:
        val_roc_auc = roc_auc_score(y_val_bin, val_proba, average='macro', multi_class='ovr')
    except ValueError as e:
        logger.warning(f"ROC AUC calculation failed for fold {fold}: {str(e)}")
        val_roc_auc = np.nan

    # Store metrics
    fold_metrics['accuracies'].append(val_acc)
    fold_metrics['precisions'].append(val_prec)
    fold_metrics['recalls'].append(val_rec)
    fold_metrics['f1s'].append(val_f1)
    fold_metrics['roc_aucs'].append(val_roc_auc)
    fold_metrics['confusion_matrices'].append(val_cm)
    fold_metrics['probabilities'].append(val_proba)
    fold_metrics['true_labels'].append(y_val)
    fold_metrics['pred_labels'].append(val_pred)
    fold_metrics['best_params'].append(grid_search.best_params_)

    logger.info(f"Fold {fold} - Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

# After all folds are processed, concatenate all fold probabilities into a single DataFrame
all_fold_probs_df = pd.concat(all_fold_probs, ignore_index=True)

# Save the aggregated probabilities to a single CSV file
all_fold_probs_df.to_csv('../../../Result/Vertiport_analysis/Probability_clustering/Training_Probabilities/all_folds_probabilities_Xgboost_with_synthetic_population.csv',
                         index=False)

logger.info("All fold probabilities have been saved to 'all_folds_probabilities_LighterModel.csv'.")

# Analyze parameter stability
param_counts = Counter(tuple(sorted(p.items())) for p in fold_metrics['best_params'])
most_common_params = param_counts.most_common()
logger.info("\nParameter Stability Analysis:")
for params, count in most_common_params:
    logger.info(f"Parameters: {dict(params)}")
    logger.info(f"Selected in {count} out of 10 folds")

# Find most common best parameters across folds
best_params = max(fold_metrics['best_params'], key=fold_metrics['best_params'].count)
logger.info(f"\nMost common best parameters across folds: {best_params}")

# Train final model on all training+validation data using most common best parameters
final_model = base_pipeline.set_params(**best_params)
final_model.fit(X_train_val, y_train_val)

# Calculate training accuracy on full training set
train_val_pred = final_model.predict(X_train_val)
train_val_acc = accuracy_score(y_train_val, train_val_pred)

# Evaluate on test set
test_pred = final_model.predict(X_test)
test_proba = final_model.predict_proba(X_test)

# Calculate per-class accuracy on test set
test_class_acc = {}
for cls in classes:
    mask = y_test == cls
    if np.any(mask):
        test_class_acc[cls] = accuracy_score(y_test[mask], test_pred[mask])
    else:
        test_class_acc[cls] = np.nan

# Save test set probabilities
test_probs_df = pd.DataFrame(test_proba, columns=classes)
test_probs_df.to_csv('../../../Result/Vertiport_analysis/Probability_clustering/Testing_Probabilities/test_set_probabilities_Xgboost_with_synthetic_population.csv',
                     index=False)

# Calculate test metrics
test_acc = accuracy_score(y_test, test_pred)
test_prec = precision_score(y_test, test_pred, average='weighted', zero_division=0)
test_rec = recall_score(y_test, test_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
test_cm = confusion_matrix(y_test, test_pred, labels=classes)

# Calculate test ROC AUC
y_test_bin = label_binarize(y_test, classes=classes)
try:
    test_roc_auc = roc_auc_score(y_test_bin, test_proba, average='macro', multi_class='ovr')
except ValueError as e:
    logger.warning(f"ROC AUC calculation failed for test set: {str(e)}")
    test_roc_auc = np.nan

# After all folds are processed, calculate and save feature importance analysis
mean_feature_importance = np.mean(fold_metrics['feature_importances'], axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_feature_importance
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
feature_importance_df.to_csv('../../../Result/Vertiport_analysis/Probability_clustering/Feature_Importance/fi_Xgboost_with_synthetic_population.csv', index=False)

# Save results
with open('../../../Result/Vertiport_analysis/Probability_clustering/Prediction_EvaluationMetrics/Xgboost_with_synthetic_population.txt', 'w') as f:
    f.write("Results for LighterModel XGBoost with 10-fold Cross-Validation:\n\n")

    # Write parameter stability analysis
    f.write("Parameter Stability Analysis:\n")
    for params, count in most_common_params:
        f.write(f"\nParameters: {dict(params)}\n")
        f.write(f"Selected in {count} out of 10 folds\n")

    f.write("\nBest Parameters per Fold:\n")
    for i, params in enumerate(fold_metrics['best_params'], 1):
        f.write(f"Fold {i}: {params}\n")
    f.write(f"\nMost Common Best Parameters: {best_params}\n\n")

    # Write overfitting analysis
    f.write("Overfitting Analysis:\n")
    f.write(
        f"Mean Training Accuracy: {np.mean(fold_metrics['train_accuracies']):.4f} ± {np.std(fold_metrics['train_accuracies']):.4f}\n")
    f.write(
        f"Mean Validation Accuracy: {np.mean(fold_metrics['accuracies']):.4f} ± {np.std(fold_metrics['accuracies']):.4f}\n")
    f.write(
        f"Training-Validation Accuracy Gap: {np.mean(fold_metrics['train_accuracies']) - np.mean(fold_metrics['accuracies']):.4f}\n\n")

    # Write per-class accuracy analysis
    f.write("Per-class Accuracy Analysis:\n")
    mean_class_acc = {}
    std_class_acc = {}
    for cls in classes:
        accs = [fold_acc[cls] for fold_acc in fold_metrics['class_accuracies'] if not np.isnan(fold_acc[cls])]
        if accs:
            mean_class_acc[cls] = np.mean(accs)
            std_class_acc[cls] = np.std(accs)
            f.write(f"Class {cls}: {mean_class_acc[cls]:.4f} ± {std_class_acc[cls]:.4f}\n")
    f.write("\n")

    f.write("Cross-validation Results (10 folds):\n")
    f.write(f"Mean Accuracy: {np.mean(fold_metrics['accuracies']):.4f} ± {np.std(fold_metrics['accuracies']):.4f}\n")
    f.write(f"Mean Precision: {np.mean(fold_metrics['precisions']):.4f} ± {np.std(fold_metrics['precisions']):.4f}\n")
    f.write(f"Mean Recall: {np.mean(fold_metrics['recalls']):.4f} ± {np.std(fold_metrics['recalls']):.4f}\n")
    f.write(f"Mean F1-score: {np.mean(fold_metrics['f1s']):.4f} ± {np.std(fold_metrics['f1s']):.4f}\n")
    f.write(f"Mean ROC AUC: {np.nanmean(fold_metrics['roc_aucs']):.4f} ± {np.nanstd(fold_metrics['roc_aucs']):.4f}\n\n")

    f.write("Per-fold Confusion Matrices:\n")
    for i, cm in enumerate(fold_metrics['confusion_matrices'], 1):
        f.write(f"\nFold {i}:\n{cm}\n")

    f.write("\nFinal ML_Model Performance:\n")
    f.write(f"Training+Validation Accuracy: {train_val_acc:.4f}\n")
    f.write(f"Test Set Accuracy: {test_acc:.4f}\n")
    f.write(f"Test-Train Accuracy Gap: {test_acc - train_val_acc:.4f}\n\n")

    f.write("Test Set Results:\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"Precision: {test_prec:.4f}\n")
    f.write(f"Recall: {test_rec:.4f}\n")
    f.write(f"F1-score: {test_f1:.4f}\n")
    f.write(f"ROC AUC: {test_roc_auc:.4f}\n")
    f.write("\nTest Set Per-class Accuracy:\n")
    for cls, acc in test_class_acc.items():
        if not np.isnan(acc):
            f.write(f"Class {cls}: {acc:.4f}\n")
    f.write("\nTest Set Confusion Matrix:\n")
    f.write(f"{test_cm}\n")

    f.write("\nTop 10 Most Important Features:\n")
    for _, row in feature_importance_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")

# Save confusion matrix
conf_matrix_df = pd.DataFrame(test_cm, index=classes, columns=classes)
conf_matrix_df.to_csv('../../../Result/Vertiport_analysis/Probability_clustering/Confusion_Matrix/Xgboost_with_synthetic_population.csv')

logger.info("Step 1 complete: ML_Model trained, validated, and tested. Results saved.")

# =========================
# 2. INITIALIZE K-MEANS++ WITH 74 VERTIPORTS
# =========================
logger.info("Step 2: Initializing k-means++ with 74 vertiports on O/D points from trial data...")
od_points = np.vstack([
    trial_data[['originX', 'originY']].values,
    trial_data[['destinationX', 'destinationY']].values
])
kmeans = KMeans(n_clusters=74, init='k-means++', random_state=RANDOM_SEED)
kmeans.fit(od_points)
vertiport_coords = kmeans.cluster_centers_
logger.info("Step 2 complete: Initial vertiport locations set.")

# =========================
# 3. ITERATIVE OPTIMIZATION
# =========================
logger.info("Step 3: Iterative vertiport optimization with UAM probability weighting...")
# Calculate car speed and cost per km from trial data
car_speed_debug_info = []
if 'tripLength-km' in trial_data.columns and 'travel time_car' in trial_data.columns:
    valid = (trial_data['travel time_car'] > 0) & (trial_data['tripLength-km'] > 0)
    filtered = trial_data[valid].copy()
    if filtered['travel time_car'].median() > 10:
        filtered['travel time_car'] = filtered['travel time_car'] / 60
    avg_car_speed = filtered['tripLength-km'].mean() / filtered['travel time_car'].mean()
    if avg_car_speed < 10 or avg_car_speed > 120 or np.isnan(avg_car_speed):
        avg_car_speed = 40
else:
    avg_car_speed = 40
if 'TravelCost_Car' in trial_data.columns and 'tripLength-km' in trial_data.columns:
    valid_cost = (trial_data['TravelCost_Car'] > 0) & (trial_data['tripLength-km'] > 0)
    filtered_cost = trial_data[valid_cost].copy()
    car_cost_per_km = (filtered_cost['TravelCost_Car'].sum() / filtered_cost['tripLength-km'].sum())
    if not (0.05 < car_cost_per_km < 2.0) or np.isnan(car_cost_per_km):
        car_cost_per_km = 0.25
else:
    car_cost_per_km = 0.25
logger.info(f"Using car speed: {avg_car_speed:.2f} km/h, car cost per km: {car_cost_per_km:.2f} €/km")

# --- Centroid history tracking ---
centroid_history = [vertiport_coords.copy()]

# UAM calculation function
VERTIPORT_K = 74
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


max_iter = 10000  # Maximum iterations to prevent infinite loops
convergence_threshold = 1e-2  # Convergence threshold for vertiport shift
converged = False
prev_coords = None
feature_cols = X_lighter.columns.tolist()

# Choose normalization method: 'simple', 'softmax', or 'log'
NORMALIZATION_METHOD = 'simple'  # Change this to test different methods
logger.info(f"Using {NORMALIZATION_METHOD} normalization within clusters")

for iteration in range(max_iter):
    logger.info(f"Iteration {iteration + 1}...")
    # a. Calculate UAM travel time and cost for each trip
    trial_with_uam = calculate_uam_time_cost(trial_data, vertiport_coords, avg_car_speed, car_cost_per_km)
    # b. Add these UAM features to the trial data (already done in trial_with_uam)
    # c. Predict mode probabilities
    for col in feature_cols: #Feature Alignment Safeguards
        if col not in trial_with_uam.columns:
            trial_with_uam[col] = 0.0
    trial_with_uam = trial_with_uam[feature_cols]
    proba = predict_mode_probabilities(trial_with_uam, final_model, feature_cols)
    # d. Use UAM probability as weights for weighted k-means
    uam_class_idx = None
    for i, cls in enumerate(classes):
        if 'uam' in str(cls).lower() or cls == 4:
            uam_class_idx = i
            break
    if uam_class_idx is None:
        uam_class_idx = len(classes) - 1
    uam_probs = proba[:, uam_class_idx]
    
    # e. Apply normalization within clusters after each clustering iteration
    # First, get current cluster assignments using previous centroids (or initial centroids for first iteration)
    from scipy.spatial.distance import cdist
    origins = trial_data[['originX', 'originY']].values
    dests = trial_data[['destinationX', 'destinationY']].values
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
centroid_history = np.array(centroid_history)  # shape: (num_iterations+1, 74, 2)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/vertiport_centroid_history.npy', centroid_history)

# =========================
# 4. FINAL PREDICTION AND OUTPUT
# =========================
logger.info("Step 4: Final prediction with optimized vertiports and saving results...")
# Calculate UAM features (full DataFrame)
trial_with_uam_full = calculate_uam_time_cost(trial_data, vertiport_coords, avg_car_speed, car_cost_per_km)
# For prediction, use only model features:
trial_with_uam = trial_with_uam_full.copy()
for col in feature_cols: #Feature Alignment Safeguards
    if col not in trial_with_uam.columns:
        trial_with_uam[col] = 0.0
trial_with_uam = trial_with_uam[feature_cols]
proba = predict_mode_probabilities(trial_with_uam, final_model, feature_cols)
output = trial_data.copy()
for i, cls in enumerate(classes):
    output[f'prob_mode_{cls}'] = proba[:, i]
# Add UAM columns from the full DataFrame
for col in ['uam_origin_vertiport', 'uam_dest_vertiport', 'travel time_Uam', 'TravelCost_Uam', 'uam_first_mile_km',
            'uam_last_mile_km', 'uam_air_km']:
    output[col] = trial_with_uam_full[col]
os.makedirs('../../../Result/Vertiport_analysis/Probability_clustering', exist_ok=True)
output.to_csv(
    '../../../Result/Vertiport_analysis/Probability_clustering/syntheticPopulation_probabilities_with_optimized_vertiports_softmax.csv',
    index=False)
np.save('../../../Result/Vertiport_analysis/Probability_clustering/Centroid/optimized_vertiport_coords_softmax.npy', vertiport_coords)
logger.info("Step 4 complete: All results saved to Result/Vertiport_analysis/Probability_clustering/")