import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# Setup logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load normalized numeric data
data = pd.read_csv('data_normalized.csv')  # Your normalized CSV path

# Clean the column names to remove special characters
data.columns = data.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)

# Define features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

# Pipeline for numeric data: impute missing values (if any)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# Original classes (for LGBM)
classes = np.unique(y)
n_classes = len(classes)

# Stratified 10-fold outer CV
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the LightGBM model and hyperparameter grid
model = LGBMClassifier(random_state=42)

# Hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [300,500],  # Single value in a list
    'classifier__max_depth': [ 3, 5],  # Adjust the tree depth
    'classifier__learning_rate': [0.01, 0.05],  # Try different learning rates
    'classifier__num_leaves': [31,50],  # Increase num_leaves to allow more splits
    'classifier__boosting_type': ['gbdt', 'dart']
}

# Create pipeline with imputer and LightGBM model
pipeline = Pipeline([
    ('imputer', numeric_transformer),
    ('classifier', model)  # Use the LightGBM model directly here
])

# Metrics storage
accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrix_sum = np.zeros((len(classes), len(classes)), dtype=int)
roc_aucs = []
all_probabilities = []
all_true_labels = []
all_pred_labels = []
importances_list = []

# Cross-validation
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    logger.info(f"LightGBM - Fold {fold} processing...")  # Replaced model_info with 'LightGBM'
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_test_bin = label_binarize(y_test, classes=classes)

    # Create a validation set for early stopping (split the training set)
    X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train the model with early stopping
    model = LGBMClassifier(random_state=42, n_estimators=50, early_stopping_rounds=20)
    model.fit(X_train_full, y_train_full,
              eval_set=[(X_val, y_val)],  # Provide the validation set for early stopping
              eval_metric='logloss')  # Provide an evaluation metric for early stopping

    # Extract feature importances
    try:
        importances = model.feature_importances_
        importances_list.append(importances)
    except AttributeError:
        # This model doesn't support feature_importances_ (e.g., SVC), so just log it
        logger.warning("Model does not support feature importances")

    # Predict on test fold
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Store probabilities and labels
    all_probabilities.append(y_proba)
    all_true_labels.append(y_test)
    all_pred_labels.append(y_pred)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr') if y_proba is not None else np.nan
    except ValueError:
        roc_auc = np.nan

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    conf_matrix_sum += cm
    roc_aucs.append(roc_auc)

    logger.info(f" Accuracy: {acc:.4f}")
    logger.info(f" Precision: {prec:.4f}")
    logger.info(f" Recall: {rec:.4f}")
    logger.info(f" F1-score: {f1:.4f}")
    logger.info(f" ROC AUC (macro): {roc_auc if not np.isnan(roc_auc) else 'N/A'}")
    logger.info(f" Confusion Matrix:\n{cm}")
    logger.info(" Interpretation:")
    logger.info("  - Diagonal values are correct predictions.")
    logger.info("  - Off-diagonal values show where the model confuses classes.\n")

logger.info(f"LightGBM - Cross-validation completed.\n")

# Aggregate feature importances over folds
mean_importances = np.mean(importances_list, axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_importances
}).sort_values(by='Importance', ascending=False)

logger.info(f"Aggregated Feature Importances:")
logger.info(feature_importance_df.head(10))

# Save the aggregated feature importances to a CSV file
feature_importance_df.to_csv('feature_importances_LightGBM.csv', index=False)
logger.info(f"Aggregated feature importances saved to 'feature_importances_LightGBM.csv'")

# Average metrics reporting
logger.info(f"LightGBM - Average Results Over 10 Folds:")
logger.info(f" Accuracy: {np.mean(accuracies):.4f}")
logger.info(f" Precision: {np.mean(precisions):.4f}")
logger.info(f" Recall: {np.mean(recalls):.4f}")
logger.info(f" F1-score: {np.mean(f1s):.4f}")
logger.info(f" ROC AUC (macro): {np.nanmean(roc_aucs):.4f}")
logger.info(f"LightGBM - Aggregated Confusion Matrix:")
logger.info(conf_matrix_sum)

# Concatenate all probabilities and labels into single arrays
all_probabilities_np = np.vstack(all_probabilities)
all_true_labels_np = np.concatenate(all_true_labels)
all_pred_labels_np = np.concatenate(all_pred_labels)

# Create a DataFrame with predicted probabilities + true/predicted labels
prob_df = pd.DataFrame(all_probabilities_np, columns=[f'prob_class_{c}' for c in classes])
prob_df['true_label'] = all_true_labels_np
prob_df['pred_label'] = all_pred_labels_np

prob_df.to_csv(f'predicted_probabilities_LightGBM.csv', index=False)
logger.info(f"LightGBM - Predicted probabilities saved to 'predicted_probabilities_LightGBM.csv'")

# Save aggregated confusion matrix to CSV
conf_matrix_df = pd.DataFrame(conf_matrix_sum, index=classes, columns=classes)
conf_matrix_df.to_csv(f'confusion_matrix_LightGBM.csv')
logger.info(f"LightGBM - Aggregated confusion matrix saved to 'confusion_matrix_LightGBM.csv'")

# Save average metrics to a text file
with open(f'average_metrics_LightGBM.txt', 'w') as f:
    f.write(f"Average Results Over 10 Folds for LightGBM:\n")
    f.write(f"Accuracy: {np.mean(accuracies):.4f}\n")
    f.write(f"Precision: {np.mean(precisions):.4f}\n")
    f.write(f"Recall: {np.mean(recalls):.4f}\n")
    f.write(f"F1-score: {np.mean(f1s):.4f}\n")
    f.write(f"ROC AUC (macro): {np.nanmean(roc_aucs):.4f}\n")

logger.info(f"LightGBM - Average metrics saved to 'average_metrics_LightGBM.txt'")
