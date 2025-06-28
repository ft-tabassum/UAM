import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
import logging
import os
from datetime import datetime
import re
import joblib


# Configure logging
def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a logger
    logger = logging.getLogger('stacking_model')
    logger.setLevel(logging.INFO)

    # Create handlers
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/stacking_model_{current_time}.log')
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logger()
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('/Result/Data_Preprocessing/Uamdata_normalized.csv')


# Clean feature names (remove special characters and spaces)
def clean_feature_names(df):
    # Create a mapping of old names to new names
    name_mapping = {}
    for col in df.columns:
        # Replace special characters and spaces with underscores
        new_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        # Remove multiple consecutive underscores
        new_name = re.sub(r'_+', '_', new_name)
        # Remove leading/trailing underscores
        new_name = new_name.strip('_')
        name_mapping[col] = new_name

    # Create a new DataFrame with cleaned column names
    df_cleaned = df.rename(columns=name_mapping)
    return df_cleaned, name_mapping


# Clean feature names
data, name_mapping = clean_feature_names(data)
logger.info("Feature names cleaned for LightGBM compatibility")

# Separate features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

logger.info("Data shapes:")
logger.info(f"Features (X): {X.shape}")
logger.info(f"Target (y): {y.shape}")

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Second split: separate validation set from remaining data (20% of original = 25% of remaining)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

logger.info("Data split sizes:")
logger.info(f"Training set: {X_train.shape[0]} samples")
logger.info(f"Validation set: {X_val.shape[0]} samples")
logger.info(f"Test set: {X_test.shape[0]} samples")

# Initialize base models
base_models = {
    'random_forest': RandomForestClassifier(random_state=42),
    'xgboost': xgb.XGBClassifier(random_state=42),
    'lightgbm': lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,  # Suppress LightGBM output
        force_col_wise=True,  # Force column-wise split
        n_jobs=-1,  # Use all available cores
        boosting_type='gbdt',  # Use traditional gradient boosting
        objective='multiclass',  # Multiclass classification
        num_class=len(np.unique(y)),  # Number of classes
        metric='multi_logloss',  # Multiclass log loss metric
        is_unbalance=True,  # Handle imbalanced data
        bagging_freq=5,  # Perform bagging every 5 iterations
        bagging_fraction=0.8  # Use 80% of data for bagging
    )
}

# Define parameter grids for each model
param_grids = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [50, 100],  # Reduced number of trees
        'max_depth': [3, 5],  # Reduced max depth
        'learning_rate': [0.01, 0.1],  # Learning rate options
        'num_leaves': [31, 63],  # Number of leaves in each tree
        'min_child_samples': [20, 50],  # Minimum samples in a leaf
        'min_child_weight': [1e-3, 1e-2],  # Minimum sum of instance weight in a leaf
        'reg_alpha': [0, 1e-1],  # L1 regularization
        'reg_lambda': [0, 1e-1],  # L2 regularization
        'min_split_gain': [0, 1e-3]  # Minimum gain for split
    }
}

# Define meta-learners and their parameter grids
meta_learners = {
    'logistic_regression': {
        'model': LogisticRegression(random_state=42),
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    },
    'svm': {
        'model': SVC(probability=True, random_state=42),
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    'neural_network': {
        'model': MLPClassifier(
            random_state=42,
            max_iter=1000,  # Increased maximum iterations
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1,  # Use 10% of data for validation
            n_iter_no_change=10,  # Stop if no improvement in 10 iterations
            tol=1e-4  # Tolerance for optimization
        ),
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01]  # Initial learning rate
        }
    }
}


# Function to validate data before training
def validate_data(X, y, model_name):
    logger.info(f"Validating data for {model_name}...")

    # Store feature names if available
    feature_names = None
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
        X = np.array(X)

    # Check for NaN values
    if np.isnan(X).any():
        logger.warning(f"NaN values found in features for {model_name}")
        X = np.nan_to_num(X, nan=np.nanmean(X))

    # Check for infinite values
    if np.isinf(X).any():
        logger.warning(f"Infinite values found in features for {model_name}")
        X = np.nan_to_num(X, nan=np.nanmean(X), posinf=np.nanmax(X), neginf=np.nanmin(X))

    # Check for constant features
    constant_features = []
    if feature_names:
        for i, col in enumerate(feature_names):
            if len(np.unique(X[:, i])) <= 1:
                constant_features.append(i)
                logger.warning(f"Constant feature found: {col}")
    else:
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) <= 1:
                constant_features.append(i)

    if constant_features:
        logger.warning(f"Removing {len(constant_features)} constant features")
        X = np.delete(X, constant_features, axis=1)
        if feature_names:
            feature_names = [f for i, f in enumerate(feature_names) if i not in constant_features]

    return X, feature_names


# Function to perform hyperparameter tuning
def tune_hyperparameters(X_train, y_train, model, param_grid, model_name):
    logger.info(f"Tuning hyperparameters for {model_name}...")
    start_time = time.time()

    # Validate data before tuning
    X_train, feature_names = validate_data(X_train, y_train, model_name)

    # For LightGBM, use a smaller number of CV folds to speed up training
    n_splits = 3 if model_name == 'lightgbm' else 5

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # For LightGBM, convert back to DataFrame with feature names
    if model_name == 'lightgbm' and feature_names is not None:
        X_train = pd.DataFrame(X_train, columns=feature_names)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Log results
    logger.info(f"Best parameters for {model_name}:")
    logger.info(grid_search.best_params_)
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

    return grid_search.best_estimator_


# Function to get base model predictions using stratified cross-validation
def get_base_predictions_cv(X_train, X_val, X_test, y_train, base_models, param_grids, n_folds=10):
    logger.info("\n" + "=" * 50)
    logger.info("Starting Base Model Training")
    logger.info("=" * 50)

    # Initialize arrays to store predictions
    train_meta_features = np.zeros((X_train.shape[0], len(base_models)))
    val_meta_features = np.zeros((X_val.shape[0], len(base_models)))
    test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Train each base model and get predictions
    for i, (name, model) in enumerate(base_models.items()):
        logger.info(f"\nTraining {name}...")
        logger.info("-" * 30)

        # Tune hyperparameters
        best_model = tune_hyperparameters(X_train, y_train, model, param_grids[name], name)

        # Get cross-validated predictions for training set
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            # Split data for this fold
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            # Train model on this fold
            best_model.fit(X_fold_train, y_fold_train)

            # Get predictions for validation set
            train_meta_features[val_idx, i] = best_model.predict_proba(X_fold_val)[:, 1]

            # Log fold performance
            fold_pred = best_model.predict(X_fold_val)
            fold_acc = accuracy_score(y_train.iloc[val_idx], fold_pred)
            fold_scores.append(fold_acc)
            logger.info(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")

        # Log average fold performance
        logger.info(f"\nAverage Fold Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

        # Train final model on full training set
        logger.info("\nTraining final model on full dataset...")
        best_model.fit(X_train, y_train)

        # Get predictions for validation and test sets
        val_meta_features[:, i] = best_model.predict_proba(X_val)[:, 1]
        test_meta_features[:, i] = best_model.predict_proba(X_test)[:, 1]

        # Log model performance using cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=n_folds, scoring='accuracy')
        logger.info(f"Final Cross-Validation Performance:")
        logger.info(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logger.info("-" * 30)

    logger.info("\n" + "=" * 50)
    logger.info("Base Model Training Completed")
    logger.info("=" * 50 + "\n")

    return train_meta_features, val_meta_features, test_meta_features


# Get base model predictions using cross-validation
logger.info("Training base models and generating meta-features using stratified cross-validation...")
train_meta_features, val_meta_features, test_meta_features = get_base_predictions_cv(
    X_train, X_val, X_test, y_train, base_models, param_grids
)


# Function to evaluate meta-learners
def evaluate_meta_learners(train_meta_features, val_meta_features, y_train, y_val):
    logger.info("\n" + "=" * 50)
    logger.info("Starting Meta-Learner Evaluation")
    logger.info("=" * 50)

    results = {}

    for name, meta_config in meta_learners.items():
        logger.info(f"\nEvaluating {name}...")
        logger.info("-" * 30)
        start_time = time.time()

        # Tune hyperparameters
        best_model = tune_hyperparameters(
            train_meta_features, y_train,
            meta_config['model'],
            meta_config['param_grid'],
            name
        )

        # Evaluate on validation set
        val_predictions = best_model.predict(val_meta_features)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # Store results
        results[name] = {
            'model': best_model,
            'accuracy': val_accuracy,
            'training_time': time.time() - start_time
        }

        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Training Time: {results[name]['training_time']:.2f} seconds")
        logger.info("-" * 30)

    logger.info("\n" + "=" * 50)
    logger.info("Meta-Learner Evaluation Completed")
    logger.info("=" * 50 + "\n")

    return results


# Evaluate all meta-learners
logger.info("Starting meta-learner evaluation...")
meta_learner_results = evaluate_meta_learners(
    train_meta_features, val_meta_features, y_train, y_val
)

# Find best meta-learner
best_meta_name = max(meta_learner_results.items(), key=lambda x: x[1]['accuracy'])[0]
best_meta_learner = meta_learner_results[best_meta_name]['model']

logger.info(f"\nBest meta-learner: {best_meta_name}")
logger.info(f"Validation Accuracy: {meta_learner_results[best_meta_name]['accuracy']:.4f}")

# Final evaluation on test set
logger.info("\nFinal Test Set Performance:")
test_predictions = best_meta_learner.predict(test_meta_features)
logger.info(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
logger.info("Classification Report:")
logger.info("\n" + classification_report(y_test, test_predictions))

# Save the model and results
model_path = 'logs/stacking_model.joblib'
joblib.dump(best_meta_learner, model_path)
logger.info(f"Model saved as '{model_path}'")

# Save meta-learner comparison results
results_df = pd.DataFrame({
    'Meta-learner': list(meta_learner_results.keys()),
    'Accuracy': [results['accuracy'] for results in meta_learner_results.values()],
    'Training Time': [results['training_time'] for results in meta_learner_results.values()]
})
results_df.to_csv('logs/meta_learner_comparison.csv', index=False)
logger.info("Meta-learner comparison results saved to 'logs/meta_learner_comparison.csv'")

