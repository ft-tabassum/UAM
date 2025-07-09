import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import Counter

# Suppress all warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load data of UAM survey data
data = pd.read_csv('../../../Result/ML_Model/Data_Preprocessing/Uamdata_normalized.csv')

# Define features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

# Store feature names
feature_names = X.columns.tolist()

# Original classes
classes = np.unique(y)
n_classes = len(classes)

# Create base pipeline
base_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('classifier', LGBMClassifier(
        random_state=42,
        verbose=-1,
        min_gain_to_split=0.1,  # Increased minimum gain for splitting
        min_data_in_leaf=30,    # Increased minimum samples in leaf
        min_sum_hessian_in_leaf=1e-2,  # Increased minimum hessian
        n_jobs=-1  # Use all available cores
    ))
])

# Define Parameter
param_grid = {
    'classifier__n_estimators': [50, 80],     # number of trees
    'classifier__max_depth': [2, 3],          # tree depth
    'classifier__learning_rate': [0.005, 0.01], # learning rate
    'classifier__num_leaves': [15, 31],       # number of leaves
    'classifier__reg_alpha': [0.1, 0.5],      #  L1 regularization
    'classifier__reg_lambda': [0.1, 0.5]      #  L2 regularization
}

# Split data into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Setup cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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
logger.info("Starting 10-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val), 1):
    logger.info(f"Processing Fold {fold}/10...")
    
    # Split data for this fold
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    # Convert to numpy arrays to avoid feature name warnings
    X_train = X_train.values
    X_val = X_val.values
    
    logger.info(f"Fold {fold} - Training data shape: {X_train.shape}")
    logger.info(f"Fold {fold} - Validation data shape: {X_val.shape}")
    
    # Use GridSearchCV for hyperparameter tuning on training data
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',  # Changed to accuracy
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    logger.info(f"Fold {fold} - Starting GridSearchCV...")
    # Fit GridSearchCV on training data
    grid_search.fit(X_train, y_train)
    logger.info(f"Fold {fold} - GridSearchCV completed")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Store feature importances
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    fold_metrics['feature_importances'].append(feature_importances)
    
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
    
    # Append the probabilities to the list
    fold_probs_df = pd.DataFrame(val_proba, columns=classes)
    fold_probs_df['fold'] = fold
    all_fold_probs.append(fold_probs_df)
    
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
    logger.info(f"Fold {fold} - Per-class Accuracy: {class_acc}")

# After all folds are processed, concatenate all fold probabilities
all_fold_probs_df = pd.concat(all_fold_probs, ignore_index=True)
all_fold_probs_df.to_csv('all_folds_probabilities_LightGBM.csv', index=False)

# Calculate and save feature importance analysis
mean_feature_importance = np.mean(fold_metrics['feature_importances'], axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_feature_importance
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
feature_importance_df.to_csv('feature_importance_LightGBM.csv', index=False)

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
final_model.fit(X_train_val.values, y_train_val)

# Calculate training accuracy on full training set
train_val_pred = final_model.predict(X_train_val.values)
train_val_acc = accuracy_score(y_train_val, train_val_pred)

# Evaluate on test set
test_pred = final_model.predict(X_test.values)
test_proba = final_model.predict_proba(X_test.values)

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
test_probs_df.to_csv('test_set_probabilities_LightGBM.csv', index=False)

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
with open('../../../Result/ML_Model/Prediction_EvaluationMetrics/Result_LightGBM.txt', 'w') as f:
    f.write("Results for LightGBM with 10-fold Cross-Validation:\n\n")
    
    # Write parameter stability analysis
    f.write("Parameter Stability Analysis:\n")
    for params, count in most_common_params:
        f.write(f"\nParameters: {dict(params)}\n")
        f.write(f"Selected in {count} out of 10 folds\n")
    
    f.write("\nBest Parameters per Fold:\n")
    for i, params in enumerate(fold_metrics['best_params'], 1):
        f.write(f"Fold {i}: {params}\n")
    f.write(f"\nMost Common Best Parameters: {best_params}\n\n")
    
    # Write feature importance analysis
    f.write("Top 10 Most Important Features:\n")
    for _, row in feature_importance_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")
    
    # Write overfitting analysis
    f.write("Overfitting Analysis:\n")
    f.write(f"Mean Training Accuracy: {np.mean(fold_metrics['train_accuracies']):.4f} ± {np.std(fold_metrics['train_accuracies']):.4f}\n")
    f.write(f"Mean Validation Accuracy: {np.mean(fold_metrics['accuracies']):.4f} ± {np.std(fold_metrics['accuracies']):.4f}\n")
    f.write(f"Training-Validation Accuracy Gap: {np.mean(fold_metrics['train_accuracies']) - np.mean(fold_metrics['accuracies']):.4f}\n\n")
    
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

# Save confusion matrix
conf_matrix_df = pd.DataFrame(test_cm, index=classes, columns=classes)
conf_matrix_df.to_csv('CM_LightGBM.csv')

logger.info("Cross-validation completed. Results saved to Result_LightGBM.txt")
