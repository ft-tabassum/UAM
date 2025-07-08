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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create output directories
os.makedirs('../../Result/Probabilities/Testing_Probabilities', exist_ok=True)
os.makedirs('../../Result/Probabilities/Training_Probabilities', exist_ok=True)
os.makedirs('../../Result/Feature_Importance', exist_ok=True)
os.makedirs('../../Result/Confusion_Matrix', exist_ok=True)
os.makedirs('../../Result/Prediction_EvaluationMetrics', exist_ok=True)

# Load LighterModel data (UAM-aware data for training)
logger.info("Loading LighterModel data...")
lighter_data = pd.read_csv('../Lightermodel/Result_LM/LighterModel_normalized.csv')

# Load trial data (UAM-unaware data for prediction)
logger.info("Loading trial data...")
trial_data = pd.read_csv('../../LargeFiles_synthetic/trial_normalized.csv')

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
    ('classifier', XGBClassifier(random_state=42))
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
    X_lighter, y_lighter, test_size=0.2, random_state=42, stratify=y_lighter
)

# Setup cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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
logger.info("Performing 10-fold cross-validation with GridSearchCV in each fold...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val), 1):
    logger.info(f"Processing Fold {fold}/10...")

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

    logger.info(f"Fold {fold} - Training Accuracy: {train_acc:.4f}")
    logger.info(f"Fold {fold} - Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Fold {fold} - Best Parameters: {grid_search.best_params_}")

# After all folds are processed, concatenate all fold probabilities into a single DataFrame
all_fold_probs_df = pd.concat(all_fold_probs, ignore_index=True)

# Save the aggregated probabilities to a single CSV file
all_fold_probs_df.to_csv('../../Result/Probabilities/Training_Probabilities/all_folds_probabilities_LighterModel.csv',
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
test_probs_df.to_csv('../../Result/Probabilities/Testing_Probabilities/test_set_probabilities_LighterModel.csv',
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
feature_importance_df.to_csv('../../Result/Feature_Importance/feature_importance_LighterModel.csv', index=False)

# Save results
with open('../../Result/Prediction_EvaluationMetrics/Result_LighterModel.txt', 'w') as f:
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

    f.write("\nFinal Model Performance:\n")
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
conf_matrix_df.to_csv('../../Result/Confusion_Matrix/CM_LighterModel.csv')

logger.info("Cross-validation completed. Results saved to Result_LighterModel.txt")

# ============================================================================
# PART 2: UAM FEATURE IMPUTATION AND TRIAL DATA PREDICTION
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("PART 2: UAM FEATURE IMPUTATION AND TRIAL DATA PREDICTION")
logger.info("=" * 80)

# Identify missing UAM features in trial data
missing_uam_features = ['travel time_CarSharing', 'TravelCost_CarSharing',
                        'travel time_RideHailing', 'TravelCost_RideHailing',
                        'travel time_Uam', 'TravelCost_Uam']

logger.info(f"Missing UAM features in trial data: {missing_uam_features}")

# Calculate UAM feature statistics from training data
uam_stats = {}
for feature in missing_uam_features:
    if feature in X_train_val.columns:
        uam_stats[feature] = {
            'mean': X_train_val[feature].mean(),
            'median': X_train_val[feature].median(),
            'std': X_train_val[feature].std()
        }
        logger.info(f"{feature}: mean={uam_stats[feature]['mean']:.4f}, "
                    f"median={uam_stats[feature]['median']:.4f}, "
                    f"std={uam_stats[feature]['std']:.4f}")

# Create three scenarios for UAM feature imputation
scenarios = {
    'Scenario_A_Zero_UAM': {feature: 0.0 for feature in missing_uam_features},
    'Scenario_B_Average_UAM': {feature: uam_stats[feature]['mean'] for feature in missing_uam_features},
    'Scenario_C_Trip_Based_UAM': {}
}

# For Scenario C, create trip-based estimates
logger.info("Creating trip-based UAM estimates for Scenario C...")
for feature in missing_uam_features:
    if feature == 'travel time_Uam':
        # UAM time: 50-70% of car time for medium-long trips, 80-90% for short trips
        scenarios['Scenario_C_Trip_Based_UAM'][feature] = 'trip_based_time'
    elif feature == 'TravelCost_Uam':
        # UAM cost: 1.5-2x car cost for short trips, 0.8-1.2x for long trips
        scenarios['Scenario_C_Trip_Based_UAM'][feature] = 'trip_based_cost'
    elif 'CarSharing' in feature:
        # Car sharing: use average values
        scenarios['Scenario_C_Trip_Based_UAM'][feature] = uam_stats[feature]['mean']
    elif 'RideHailing' in feature:
        # Ride hailing: use average values
        scenarios['Scenario_C_Trip_Based_UAM'][feature] = uam_stats[feature]['mean']

# Prepare trial data for prediction
logger.info("Preparing trial data for prediction...")

# Create a copy of trial data for each scenario
trial_predictions = {}

for scenario_name, uam_values in scenarios.items():
    logger.info(f"\nProcessing {scenario_name}...")

    # Create a copy of trial data
    trial_scenario = trial_data.copy()

    # Add missing UAM features
    for feature, value in uam_values.items():
        if isinstance(value, str) and value.startswith('trip_based'):
            # Calculate trip-based estimates
            if value == 'trip_based_time':
                # UAM time: 60% of car time for trips > 10km, 80% for shorter trips
                car_time_col = 'travel time_car'
                if car_time_col in trial_scenario.columns:
                    trial_scenario[feature] = np.where(
                        trial_scenario['tripLength-km'] > 10,
                        trial_scenario[car_time_col] * 0.6,
                        trial_scenario[car_time_col] * 0.8
                    )
                else:
                    trial_scenario[feature] = uam_stats[feature]['mean']
            elif value == 'trip_based_cost':
                # UAM cost: 1.8x car cost for trips < 5km, 1.2x for 5-15km, 0.9x for >15km
                car_cost_col = 'TravelCost_Car'
                if car_cost_col in trial_scenario.columns:
                    trial_scenario[feature] = np.where(
                        trial_scenario['tripLength-km'] < 5,
                        trial_scenario[car_cost_col] * 1.8,
                        np.where(
                            trial_scenario['tripLength-km'] < 15,
                            trial_scenario[car_cost_col] * 1.2,
                            trial_scenario[car_cost_col] * 0.9
                        )
                    )
                else:
                    trial_scenario[feature] = uam_stats[feature]['mean']
        else:
            # Use fixed value
            trial_scenario[feature] = value

    # Ensure all features are in the same order as training data
    missing_features = set(X_train_val.columns) - set(trial_scenario.columns)
    for feature in missing_features:
        trial_scenario[feature] = 0.0

    # Reorder columns to match training data
    trial_scenario = trial_scenario[X_train_val.columns]

    # Make predictions
    trial_proba = final_model.predict_proba(trial_scenario)

    # Create results DataFrame
    trial_results = pd.DataFrame(trial_proba, columns=classes)

    # Add original trial data features
    for col in trial_data.columns:
        if col not in trial_results.columns:
            trial_results[col] = trial_data[col].values

    # Add scenario information
    trial_results['scenario'] = scenario_name

    # Store results
    trial_predictions[scenario_name] = trial_results

    # Save scenario-specific results
    trial_results.to_csv(f'../../Result/Probabilities/Testing_Probabilities/trial_predictions_{scenario_name}.csv',
                         index=False)

    # Calculate summary statistics
    uam_class_index = None
    for i, cls in enumerate(classes):
        if 'uam' in str(cls).lower() or cls == 4:  # Assuming UAM is class 4 based on the data
            uam_class_index = i
            break

    if uam_class_index is not None:
        uam_probabilities = trial_proba[:, uam_class_index]
        logger.info(f"{scenario_name} UAM Statistics:")
        logger.info(f"  Mean UAM probability: {np.mean(uam_probabilities):.4f}")
        logger.info(f"  Median UAM probability: {np.median(uam_probabilities):.4f}")
        logger.info(f"  Std UAM probability: {np.std(uam_probabilities):.4f}")
        logger.info(f"  Max UAM probability: {np.max(uam_probabilities):.4f}")
        logger.info(f"  Min UAM probability: {np.min(uam_probabilities):.4f}")
        logger.info(f"  Trips with UAM probability > 0.5: {np.sum(uam_probabilities > 0.5)}")
        logger.info(f"  Trips with UAM probability > 0.3: {np.sum(uam_probabilities > 0.3)}")

# Create comprehensive comparison
logger.info("\nCreating comprehensive comparison...")
comparison_data = []

for scenario_name, trial_results in trial_predictions.items():
    # Find UAM class
    uam_class_index = None
    for i, cls in enumerate(classes):
        if 'uam' in str(cls).lower() or cls == 4:
            uam_class_index = i
            break

    if uam_class_index is not None:
        uam_probabilities = trial_results.iloc[:, uam_class_index]

        # Add trip characteristics
        comparison_df = pd.DataFrame({
            'scenario': scenario_name,
            'trip_id': trial_results['trip_id'],
            'tripLength-km': trial_results['tripLength-km'],
            'travel_time_car': trial_results['travel time_car'],
            'TravelCost_Car': trial_results['TravelCost_Car'],
            'purpose': trial_results['purpose'],
            'age': trial_results['age'],
            'gender': trial_results['gender'],
            'Monthly_Income': trial_results['Monthly_Income'],
            'uam_probability': uam_probabilities,
            'predicted_mode': trial_results.iloc[:, :len(classes)].idxmax(axis=1)
        })

        comparison_data.append(comparison_df)

# Combine all scenarios
if comparison_data:
    comprehensive_comparison = pd.concat(comparison_data, ignore_index=True)
    comprehensive_comparison.to_csv('../../Result/Probabilities/Testing_Probabilities/comprehensive_uam_analysis.csv',
                                    index=False)

    # Create summary analysis
    summary_analysis = comprehensive_comparison.groupby('scenario')['uam_probability'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max',
        lambda x: np.sum(x > 0.3),
        lambda x: np.sum(x > 0.5),
        lambda x: np.sum(x > 0.7)
    ]).rename(columns={
        '<lambda_0>': 'trips_uam_prob_>0.3',
        '<lambda_1>': 'trips_uam_prob_>0.5',
        '<lambda_2>': 'trips_uam_prob_>0.7'
    })

    summary_analysis.to_csv('../../Result/Probabilities/Testing_Probabilities/uam_probability_summary.csv')

    # Analyze high-potential trips
    high_potential_trips = comprehensive_comparison[comprehensive_comparison['uam_probability'] > 0.5]
    high_potential_trips.to_csv('../../Result/Probabilities/Testing_Probabilities/high_uam_potential_trips.csv',
                                index=False)

    # Trip length analysis
    trip_length_analysis = \
    comprehensive_comparison.groupby(['scenario', pd.cut(comprehensive_comparison['tripLength-km'],
                                                         bins=[0, 5, 10, 15, 20, 50, 100])])['uam_probability'].mean()
    trip_length_analysis.to_csv('../../Result/Probabilities/Testing_Probabilities/uam_probability_by_trip_length.csv')

    logger.info("\nUAM Analysis Summary:")
    logger.info(
        f"Total trips analyzed: {len(comprehensive_comparison[comprehensive_comparison['scenario'] == 'Scenario_A_Zero_UAM'])}")
    logger.info(f"High potential trips (>50% UAM probability): {len(high_potential_trips)}")

    # Print scenario comparisons
    for scenario in ['Scenario_A_Zero_UAM', 'Scenario_B_Average_UAM', 'Scenario_C_Trip_Based_UAM']:
        scenario_data = comprehensive_comparison[comprehensive_comparison['scenario'] == scenario]
        logger.info(f"\n{scenario}:")
        logger.info(f"  Mean UAM probability: {scenario_data['uam_probability'].mean():.4f}")
        logger.info(f"  High potential trips: {len(scenario_data[scenario_data['uam_probability'] > 0.5])}")

logger.info("\nUAM Analysis completed successfully!")
logger.info("All results saved to Result/Probabilities/Testing_Probabilities/")