import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Setup logging (optional, replace with print if )
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load normalized numeric data
data = pd.read_csv('data_normalized.csv')

# Define features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

# Check if any categorical columns remain (optional)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    logger.warning(f"Categorical columns found but this code assumes numeric only: {categorical_cols}")
    #print("Warning: Categorical columns found, but this code assumes numeric only.")

# Pipeline for numeric data: impute missing values (if any)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# Complete pipeline: preprocessing + classifier
pipeline = Pipeline([
    ('imputer', numeric_transformer),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__bootstrap': [True, False]
}

# Classes for ROC AUC binarization
classes = np.unique(y)
n_classes = len(classes)

# Stratified 10-fold outer CV
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Metrics storage
accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrix_sum = np.zeros((n_classes, n_classes), dtype=int)
roc_aucs = []
all_probabilities = []
all_true_labels = []
all_pred_labels = []
importances_list = []

logger.info("Starting cross-validation with hyperparameter tuning...\n")

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    logger.info(f"Fold {fold} processing...")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_test_bin = label_binarize(y_test, classes=classes)

    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f" Best params: {grid_search.best_params_}")

    # Extract and store feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    importances_list.append(importances)

    # Predict on test fold
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

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

    # ROC AUC (macro, multi-class OvR)
    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    except ValueError:
        roc_auc = np.nan

    # Append results
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    conf_matrix_sum += cm
    roc_aucs.append(roc_auc)

    # Fold results logging
    logger.info(f" Accuracy: {acc:.4f}")
    logger.info(f" Precision: {prec:.4f}")
    logger.info(f" Recall: {rec:.4f}")
    logger.info(f" F1-score: {f1:.4f}")
    logger.info(f" ROC AUC (macro): {roc_auc if not np.isnan(roc_auc) else 'N/A'}")
    logger.info(f" Confusion Matrix:\n{cm}")
    logger.info(" Interpretation:")
    logger.info("  - Diagonal values are correct predictions.")
    logger.info("  - Off-diagonal values show where the model confuses classes.\n")

logger.info("Cross-validation completed.\n")

# Aggregate feature importances over folds
mean_importances = np.mean(importances_list, axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': mean_importances
}).sort_values(by='importance', ascending=False)

logger.info("Aggregated Feature Importances:")
logger.info(feature_importance_df.head(10))

# Save feature importance
feature_importance_df.to_csv('feature_importances.csv', index=False)
logger.info("Feature importances saved to 'feature_importances.csv'")

# Average metrics reporting
logger.info("Average Results Over 10 Folds:")
logger.info(f" Accuracy: {np.mean(accuracies):.4f}")
logger.info(f" Precision: {np.mean(precisions):.4f}")
logger.info(f" Recall: {np.mean(recalls):.4f}")
logger.info(f" F1-score: {np.mean(f1s):.4f}")
logger.info(f" ROC AUC (macro): {np.nanmean(roc_aucs):.4f}")
logger.info("Aggregated Confusion Matrix:")
logger.info(conf_matrix_sum)

# Concatenate all probabilities and labels into single arrays
all_probabilities = np.vstack(all_probabilities)
all_true_labels = np.concatenate(all_true_labels)
all_pred_labels = np.concatenate(all_pred_labels)

# Create a DataFrame with predicted probabilities + true/predicted labels
prob_df = pd.DataFrame(all_probabilities, columns=[f'prob_class_{c}' for c in classes])
prob_df['true_label'] = all_true_labels
prob_df['pred_label'] = all_pred_labels

logger.info("\nExample of predicted probabilities DataFrame head:")
logger.info(prob_df.head())

# Save predicted probabilities and labels to CSV
prob_df.to_csv('predicted_probabilities.csv', index=False)
logger.info("Predicted probabilities saved to 'predicted_probabilities.csv'")

# Save aggregated confusion matrix to CSV
conf_matrix_df = pd.DataFrame(conf_matrix_sum, index=classes, columns=classes)
conf_matrix_df.to_csv('confusion_matrix.csv')
logger.info("Aggregated confusion matrix saved to 'confusion_matrix.csv'")

# Save average metrics to a text file
with open('average_metrics.txt', 'w') as f:
    f.write("Average Results Over 10 Folds:\n")
    f.write(f"Accuracy: {np.mean(accuracies):.4f}\n")
    f.write(f"Precision: {np.mean(precisions):.4f}\n")
    f.write(f"Recall: {np.mean(recalls):.4f}\n")
    f.write(f"F1-score: {np.mean(f1s):.4f}\n")
    f.write(f"ROC AUC (macro): {np.nanmean(roc_aucs):.4f}\n")

#logger.info("Average metrics saved to 'average_metrics.txt'")