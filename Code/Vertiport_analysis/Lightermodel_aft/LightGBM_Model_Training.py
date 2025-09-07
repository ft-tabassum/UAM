import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import Counter
import os
import pickle

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Custom imputer that preserves feature names
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

# Create output directories
base_path = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_LightGBM"
os.makedirs(f"{base_path}/Trained_Model_EvaluationOutput/Testing_Probabilities", exist_ok=True)
os.makedirs(f"{base_path}/Trained_Model_EvaluationOutput/Training_Probabilities", exist_ok=True)
os.makedirs(f"{base_path}/Trained_Model_EvaluationOutput/Feature_Importance", exist_ok=True)
os.makedirs(f"{base_path}/Trained_Model_EvaluationOutput/Confusion_Matrix", exist_ok=True)
os.makedirs(f"{base_path}/Trained_Model_EvaluationOutput/Prediction_EvaluationMetrics", exist_ok=True)
os.makedirs(f"{base_path}/Trained_Model_LightGBM", exist_ok=True)

# Load LighterModel data (UAM-aware data for training)
logger.info("Loading processed LighterModel data...")
lighter_data = pd.read_csv("D:/Thesis/UAM/Result/Vertiport_analysis/Model_LightGBM/Lighter_LGBM_DataPreprocessing/LighterModelProcessing_aft.csv")

# Define features and target for LighterModel
y_lighter = lighter_data['CHOICE']
X_lighter = lighter_data.drop(columns=['CHOICE'])

# Original classes
classes = np.unique(y_lighter)
n_classes = len(classes)

# Define class names
class_names = {
    0: 'Car',
    1: 'Public Transport', 
    2: 'Autonomous Flying Taxi'
}

logger.info(f"Number of classes: {n_classes}")
logger.info(f"Classes: {classes}")
logger.info("Class mapping:")
for class_num, class_name in class_names.items():
    logger.info(f"  {class_num}: {class_name}")

# Create base pipeline
base_pipeline = Pipeline([
    ('imputer', FeaturePreservingImputer(strategy='constant', fill_value=0)),
    ('classifier', LGBMClassifier(
        random_state=RANDOM_SEED,
        verbose=-1,
        min_gain_to_split=0.05,
        min_data_in_leaf=20,
        min_sum_hessian_in_leaf=1e-3,
        class_weight='balanced',
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1
    ))
])

# Hyperparameter grid - LightGBM specific parameters
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__num_leaves': [63, 127],
    'classifier__reg_alpha': [0.1, 0.5],
    'classifier__reg_lambda': [0.1, 0.5]
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
logger.info("Step 1: Training, validating, and testing LightGBM model on LighterModel data...")
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
    
    # Log per-class accuracy with class names
    logger.info(f"Fold {fold} - Per-class Accuracy:")
    for cls in classes:
        if cls in class_names:
            logger.info(f"  {class_names[cls]}: {class_acc[cls]:.4f}")
        else:
            logger.info(f"  Class {cls}: {class_acc[cls]:.4f}")

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
all_fold_probs_df.to_csv(f'{base_path}/Trained_Model_EvaluationOutput/Training_Probabilities/all_folds_probabilities_LightGBM_LighterModel_training.csv',
                         index=False)

logger.info("All fold probabilities have been saved to 'all_folds_probabilities_LightGBM_LighterModel_training.csv'.")

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

# Log test set per-class accuracy with class names
logger.info("Test Set Per-class Accuracy:")
for cls in classes:
    if cls in class_names:
        logger.info(f"  {class_names[cls]}: {test_class_acc[cls]:.4f}")
    else:
        logger.info(f"  Class {cls}: {test_class_acc[cls]:.4f}")

# Save test set probabilities
test_probs_df = pd.DataFrame(test_proba, columns=classes)
test_probs_df.to_csv(f'{base_path}/Trained_Model_EvaluationOutput/Testing_Probabilities/test_set_probabilities_LightGBM_LighterModel_testing.csv',
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
feature_importance_df.to_csv(f'{base_path}/Trained_Model_EvaluationOutput/Feature_Importance/fi_LightGBM_LighterModel.csv', index=False)

# Save results
with open(f'{base_path}/Trained_Model_EvaluationOutput/Prediction_EvaluationMetrics/LightGBM_LighterModel_evaluation.txt', 'w') as f:
    f.write("Results for LighterModel LightGBM with 10-fold Cross-Validation:\n\n")

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
            class_name = class_names.get(cls, f"Class {cls}")
            f.write(f"{class_name}: {mean_class_acc[cls]:.4f} ± {std_class_acc[cls]:.4f}\n")
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

    f.write("\nFinal LighterModel Performance:\n")
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
            class_name = class_names.get(cls, f"Class {cls}")
            f.write(f"{class_name}: {acc:.4f}\n")
    f.write("\nTest Set Confusion Matrix:\n")
    f.write(f"{test_cm}\n")

    f.write("\nTop 10 Most Important Features:\n")
    for _, row in feature_importance_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")

# Save confusion matrix with class names
conf_matrix_df = pd.DataFrame(test_cm, 
                             index=[class_names.get(cls, f"Class {cls}") for cls in classes], 
                             columns=[class_names.get(cls, f"Class {cls}") for cls in classes])
conf_matrix_df.to_csv(f'{base_path}/Trained_Model_EvaluationOutput/Confusion_Matrix/LightGBM_LighterModel_confusion_matrix.csv')

# Save the trained model and related data for Part 2
model_data = {
    'final_model': final_model,
    'feature_names': feature_names,
    'classes': classes,
    'class_names': class_names,  # Add class names for Part 2
    'best_params': best_params,
    'test_acc': test_acc,
    'test_prec': test_prec,
    'test_rec': test_rec,
    'test_f1': test_f1,
    'test_roc_auc': test_roc_auc
}
with open(f'{base_path}/Trained_Model_LightGBM/lightgbm_model_LighterModel.pkl', 'wb') as f:
    pickle.dump(model_data, f)

logger.info("Step 1 complete: LighterModel trained, validated, and tested. Model saved for Part 2.")
logger.info(f"Model saved to: {base_path}/Trained_Model_LightGBM/lightgbm_model_LighterModel.pkl")
