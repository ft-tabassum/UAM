import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb
import warnings
import logging
import os
from datetime import datetime
import joblib
import re
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Setup logging
def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger('stacking_svm')
    logger.setLevel(logging.INFO)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/stacking_svm_{current_time}.log')
    console_handler = logging.StreamHandler()

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logger()
warnings.filterwarnings('ignore')

# Load data of UAM survey data
logger.info("Loading data...")
data = pd.read_csv('D:/Thesis/UAM/Result/ML_Model/Data_Preprocessing/Uamdata_normalized.csv')


# Clean feature names to remove special characters
def clean_feature_names(df):
    # Create a copy of the dataframe
    df_clean = df.copy()

    # Log original column names
    logger.info("Original column names:")
    logger.info(df_clean.columns.tolist())

    # More thorough cleaning of column names
    def clean_name(name):
        # Replace special characters with underscore
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name

    # Clean column names
    df_clean.columns = [clean_name(col) for col in df_clean.columns]

    # Log cleaned column names
    logger.info("Cleaned column names:")
    logger.info(df_clean.columns.tolist())

    return df_clean


# Clean feature names
data = clean_feature_names(data)

# Separate features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

# Get original classes
classes = np.unique(y)
n_classes = len(classes)

logger.info("Data shapes:")
logger.info(f"Features (X): {X.shape}")
logger.info(f"Target (y): {y.shape}")

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_SEED,
                                                  stratify=y_train_val)

logger.info("Data split sizes:")
logger.info(f"Training set: {X_train.shape[0]} samples")
logger.info(f"Validation set: {X_val.shape[0]} samples")
logger.info(f"Test set: {X_test.shape[0]} samples")

# Define parameter grids for base models (only important parameters)
param_grids = {
    'lightgbm': {
        'n_estimators': [80, 100],  # 2 values
        'max_depth': [8, 10],  # 2 values
        'learning_rate': [0.05, 0.1]  # 2 values
    },
    'random_forest': {
        'n_estimators': [70, 80, 90],
        'max_depth': [8, 10, 12],
        'min_samples_split': [5, 8, 10],
        'min_samples_leaf': [2, 3, 4],
        'max_features': ['sqrt']
    },
    'xgboost': {
        'n_estimators': [70, 80, 90],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
}

# Initialize base models
base_models = {
    'lightgbm': lgb.LGBMClassifier(
        random_state=RANDOM_SEED,
        verbose=-1,
        force_col_wise=True,
        n_jobs=-1,
        boosting_type='gbdt',
        objective='multiclass',
        num_class=len(classes),
        metric='multi_logloss',
        is_unbalance=True,
        bagging_freq=5,
        bagging_fraction=0.8,
        min_child_weight=1e-3,
        min_split_gain=0
    ),
    'random_forest': RandomForestClassifier(random_state=RANDOM_SEED),
    'xgboost': xgb.XGBClassifier(random_state=RANDOM_SEED)
}

# Initialize storage for metrics
fold_metrics = {
    'accuracies': [], 'precisions': [], 'recalls': [],
    'f1s': [], 'roc_aucs': [], 'confusion_matrices': [],
    'probabilities': [], 'true_labels': [], 'pred_labels': [],
    'best_params': [], 'train_accuracies': [], 'class_accuracies': [],
    'feature_importances': []
}


# Function to get base model predictions using stratified cross-validation
def get_base_predictions_cv(X_train, X_val, X_test, y_train, base_models, param_grids, n_folds=10):
    logger.info("\n" + "=" * 50)
    logger.info("Starting Base ML_Model_survey_data Training")
    logger.info("=" * 50)

    train_meta_features = np.zeros((X_train.shape[0], len(base_models)))
    val_meta_features = np.zeros((X_val.shape[0], len(base_models)))
    test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

    # Dictionary to store best parameters for each model
    best_params_dict = {}
    all_fold_params = {name: [] for name in base_models.keys()}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for i, (name, model) in enumerate(base_models.items()):
        logger.info(f"\nTraining {name}...")
        logger.info("-" * 30)

        # GridSearchCV for all models including LightGBM
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        # Get cross-validated predictions
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            grid_search.fit(X_fold_train, y_fold_train)
            best_model = grid_search.best_estimator_
            train_meta_features[val_idx, i] = best_model.predict_proba(X_fold_val)[:, 1]
            fold_pred = best_model.predict(X_fold_val)
            fold_acc = accuracy_score(y_train.iloc[val_idx], fold_pred)
            fold_scores.append(fold_acc)

            logger.info(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
            logger.info(f"Best Parameters: {grid_search.best_params_}")

            # Store best parameters for this fold
            all_fold_params[name].append(grid_search.best_params_)

        # Train final model
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Calculate and log final performance
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=n_folds, scoring='accuracy')
        logger.info(f"\nMean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Store best parameters
        best_params_dict[name] = grid_search.best_params_

        logger.info("-" * 30)

        # Get predictions for validation and test sets
        val_meta_features[:, i] = best_model.predict_proba(X_val)[:, 1]
        test_meta_features[:, i] = best_model.predict_proba(X_test)[:, 1]

    return train_meta_features, val_meta_features, test_meta_features, best_params_dict, all_fold_params


# Get base model predictions
logger.info("Training base models and generating meta-features...")
train_meta_features, val_meta_features, test_meta_features, best_params_dict, all_fold_params = get_base_predictions_cv(
    X_train, X_val, X_test, y_train, base_models, param_grids
)

# Define SVM meta-learner with important parameters only
svm_meta_learner = SVC(probability=True, random_state=RANDOM_SEED)
svm_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

# Train SVM meta-learner
logger.info("\nTraining SVM meta-learner...")
grid_search = GridSearchCV(
    estimator=svm_meta_learner,
    param_grid=svm_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(train_meta_features, y_train)
best_svm = grid_search.best_estimator_

# Store SVM parameters
best_params_dict['svm_meta_learner'] = grid_search.best_params_

# Evaluate on validation set
val_predictions = best_svm.predict(val_meta_features)
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions, average='weighted')
val_recall = recall_score(y_val, val_predictions, average='weighted')
val_f1 = f1_score(y_val, val_predictions, average='weighted')

logger.info("\nValidation Set Performance:")
logger.info(f"Accuracy: {val_accuracy:.4f}")
logger.info(f"Precision: {val_precision:.4f}")
logger.info(f"Recall: {val_recall:.4f}")
logger.info(f"F1 Score: {val_f1:.4f}")
logger.info("\nBest SVM Parameters:")
logger.info(grid_search.best_params_)

# Final evaluation on test set
test_predictions = best_svm.predict(test_meta_features)
test_proba = best_svm.predict_proba(test_meta_features)

# Calculate test metrics
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='weighted')
test_recall = recall_score(y_test, test_predictions, average='weighted')
test_f1 = f1_score(y_test, test_predictions, average='weighted')
test_cm = confusion_matrix(y_test, test_predictions)

# Calculate ROC AUC
y_test_bin = label_binarize(y_test, classes=classes)
test_roc_auc = roc_auc_score(y_test_bin, test_proba, average='macro', multi_class='ovr')

# Calculate class-wise metrics
class_precision = precision_score(y_test, test_predictions, average=None)
class_recall = recall_score(y_test, test_predictions, average=None)
class_f1 = f1_score(y_test, test_predictions, average=None)

# Calculate feature importances for base models
feature_importances = {}
for name, model in base_models.items():
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_

# Save results to text file
with open('D:/Thesis/UAM/Result/ML_Model/Prediction_EvaluationMetrics/Result_Stacking.txt', 'w') as f:
    f.write("Results for Stacking with SVM Meta-learner (10-fold Cross-Validation):\n\n")

    f.write("Parameter Stability Analysis:\n\n")

    # Analyze parameter stability for each model
    for model_name, fold_params in all_fold_params.items():
        if fold_params:  # Skip if empty
            f.write(f"{model_name.upper()} Parameters:\n")
            # Count parameter combinations
            param_counts = {}
            for params in fold_params:
                param_str = str(params)
                param_counts[param_str] = param_counts.get(param_str, 0) + 1

            # Sort by count
            sorted_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)

            for param_str, count in sorted_params:
                f.write(f"Parameters: {param_str}\n")
                f.write(f"Selected in {count} out of 10 folds\n\n")

    f.write("Best Parameters per Fold:\n")
    for fold in range(10):
        f.write(f"Fold {fold + 1}:\n")
        for model_name, fold_params in all_fold_params.items():
            if fold_params and fold < len(fold_params):
                f.write(f"  {model_name}: {fold_params[fold]}\n")
        f.write("\n")

    f.write("Most Common Best Parameters:\n")
    for model_name, fold_params in all_fold_params.items():
        if fold_params:
            param_counts = {}
            for params in fold_params:
                param_str = str(params)
                param_counts[param_str] = param_counts.get(param_str, 0) + 1
            most_common = max(param_counts.items(), key=lambda x: x[1])
            f.write(f"{model_name}: {most_common[0]}\n")
    f.write("\n")

    f.write("Overfitting Analysis:\n")
    f.write(f"Mean Training Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Mean Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Training-Validation Accuracy Gap: {0:.4f}\n\n")

    f.write("Per-class Accuracy Analysis:\n")
    for i, class_name in enumerate(classes):
        f.write(f"Class {i}: {class_recall[i]:.4f}\n")
    f.write("\n")

    f.write("Cross-validation Results (10 folds):\n")
    f.write(f"Mean Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Mean Precision: {val_precision:.4f}\n")
    f.write(f"Mean Recall: {val_recall:.4f}\n")
    f.write(f"Mean F1-score: {val_f1:.4f}\n")
    f.write(f"Mean ROC AUC: {test_roc_auc:.4f}\n\n")

    f.write("Final ML_Model_survey_data Performance:\n")
    f.write(f"Training+Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Test Set Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test-Train Accuracy Gap: {test_accuracy - val_accuracy:.4f}\n\n")

    f.write("Test Set Results:\n")
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"F1-score: {test_f1:.4f}\n")
    f.write(f"ROC AUC: {test_roc_auc:.4f}\n\n")

    f.write("Test Set Per-class Accuracy:\n")
    for i, class_name in enumerate(classes):
        f.write(f"Class {i}: {class_recall[i]:.4f}\n")
    f.write("\n")

    f.write("Test Set Confusion Matrix:\n")
    f.write(str(test_cm))
    f.write("\n\n")

    # Feature importances for each base model
    for name, importances in feature_importances.items():
        f.write(f"Top 10 Most Important Features ({name.upper()}):\n")
        # Get top 10 features
        top_indices = np.argsort(importances)[-10:][::-1]
        for idx in top_indices:
            f.write(f"{X.columns[idx]}: {importances[idx]:.4f}\n")
        f.write("\n")

logger.info("Results saved to 'Result_Stacking.txt'")

# Save model
model_path = 'D:/Thesis/UAM/Result/ML_Model/Probabilities/Testing_Probabilities/stacking_svm_model.joblib'
joblib.dump(best_svm, model_path)
logger.info(f"\nML_Model_survey_data saved as '{model_path}'")

# Save test set probabilities
test_probs_df = pd.DataFrame(test_proba, columns=classes)
test_probs_df.to_csv('D:/Thesis/UAM/Result/ML_Model_survey_data/Probabilities/Testing_Probabilities/stacking_svm_test_probabilities.csv', index=False)
logger.info("Test set probabilities saved to 'stacking_svm_test_probabilities.csv'")

# Save confusion matrix
confusion_matrix_df = pd.DataFrame(test_cm,
                                   index=[f'True_{c}' for c in classes],
                                   columns=[f'Pred_{c}' for c in classes])
confusion_matrix_df.to_csv('D:/Thesis/UAM/Result/ML_Model_survey_data/Confusion_Matrix/CM_Stacking.csv', index=True)
logger.info("Confusion matrix saved to 'CM_Stacking.csv'")

# Save feature importances
for name, importances in feature_importances.items():
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    feature_importance_df.to_csv(
        f'D:/Thesis/UAM/Result/ML_Model_survey_data/Feature_Importance/stacking_svm_{name}_feature_importance.csv', index=False)
    logger.info(f"Feature importances for {name} saved to 'stacking_svm_{name}_feature_importance.csv'")+
    