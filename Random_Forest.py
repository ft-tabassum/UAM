import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load data
data = pd.read_csv('data_normalized.csv')

# Define features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

# Original classes
classes = np.unique(y)
n_classes = len(classes)

# Create base pipeline
base_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'classifier__n_estimators': [65, 70, 72, 75],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
}

# Split data into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# First, use GridSearchCV to find best hyperparameters
logger.info("Finding best hyperparameters using GridSearchCV...")
grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=10,  # Use 10-fold CV for hyperparameter tuning
    scoring='accuracy',
    n_jobs=-1
)

# Fit GridSearchCV on training+validation data
grid_search.fit(X_train_val, y_train_val)

# Get best parameters
best_params = grid_search.best_params_
logger.info(f"Best parameters found: {best_params}")

# Create new pipeline with best parameters
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('classifier', RandomForestClassifier(**{k.split('__')[1]: v for k, v in best_params.items()}, random_state=42))
])

# Setup cross-validation for model evaluation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize storage for metrics
fold_metrics = {
    'accuracies': [], 'precisions': [], 'recalls': [],
    'f1s': [], 'roc_aucs': [], 'confusion_matrices': [],
    'probabilities': [], 'true_labels': [], 'pred_labels': []
}
# Initialize a list to store the probabilities
all_fold_probs = []

# Perform cross-validation with best parameters
logger.info("Performing 10-fold cross-validation with best parameters...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val), 1):
    logger.info(f"Processing Fold {fold}/10...")
    
    # Split data for this fold
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    # Train model with best parameters
    pipeline.fit(X_train, y_train)
    
    # Make predictions on validation set
    val_pred = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)

    # Append the probabilities to the list (add fold number as a column)
    fold_probs_df = pd.DataFrame(val_proba, columns=classes)
    fold_probs_df['fold'] = fold  # Add fold number to distinguish rows
    all_fold_probs.append(fold_probs_df)

    # Save fold probabilities to a CSV file after each fold
    #fold_probs_df = pd.DataFrame(val_proba, columns=classes)
    #fold_probs_df.to_csv(f'fold_{fold}_probabilities_RandomForest.csv', index=False)
    
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
    
    logger.info(f"Fold {fold} - Validation Accuracy: {val_acc:.4f}")

# After all folds are processed, concatenate all fold probabilities into a single DataFrame
all_fold_probs_df = pd.concat(all_fold_probs, ignore_index=True)

# Save the aggregated probabilities to a single CSV file
all_fold_probs_df.to_csv('all_folds_probabilities_RandomForest.csv', index=False)

logger.info("All fold probabilities have been saved to 'all_folds_probabilities_RandomForest.csv'.")

# Train final model on all training+validation data with best parameters
final_model = pipeline.fit(X_train_val, y_train_val)

# Evaluate on test set
test_pred = final_model.predict(X_test)
test_proba = final_model.predict_proba(X_test)

# Save test set probabilities
test_probs_df = pd.DataFrame(test_proba, columns=classes)
test_probs_df.to_csv('test_set_probabilities_RandomForest.csv', index=False)

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

# Save results
with open('Result_RandomForest.txt', 'w') as f:
    f.write("Results for RandomForest with 10-fold Cross-Validation:\n\n")
    f.write(f"Best Parameters Found: {best_params}\n\n")
    f.write("Cross-validation Results (10 folds):\n")
    f.write(f"Mean Accuracy: {np.mean(fold_metrics['accuracies']):.4f} ± {np.std(fold_metrics['accuracies']):.4f}\n")
    f.write(f"Mean Precision: {np.mean(fold_metrics['precisions']):.4f} ± {np.std(fold_metrics['precisions']):.4f}\n")
    f.write(f"Mean Recall: {np.mean(fold_metrics['recalls']):.4f} ± {np.std(fold_metrics['recalls']):.4f}\n")
    f.write(f"Mean F1-score: {np.mean(fold_metrics['f1s']):.4f} ± {np.std(fold_metrics['f1s']):.4f}\n")
    f.write(f"Mean ROC AUC: {np.nanmean(fold_metrics['roc_aucs']):.4f} ± {np.nanstd(fold_metrics['roc_aucs']):.4f}\n\n")
    
    f.write("Per-fold Confusion Matrices:\n")
    for i, cm in enumerate(fold_metrics['confusion_matrices'], 1):
        f.write(f"\nFold {i}:\n{cm}\n")
    
    f.write("\nTest Set Results:\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"Precision: {test_prec:.4f}\n")
    f.write(f"Recall: {test_rec:.4f}\n")
    f.write(f"F1-score: {test_f1:.4f}\n")
    f.write(f"ROC AUC: {test_roc_auc:.4f}\n")
    f.write("\nTest Set Confusion Matrix:\n")
    f.write(f"{test_cm}\n")

# Save confusion matrix
conf_matrix_df = pd.DataFrame(test_cm, index=classes, columns=classes)
conf_matrix_df.to_csv('CM_confusion_matrix_RandomForest.csv')

logger.info("Cross-validation completed. Results saved to Result_RandomForest.txt")