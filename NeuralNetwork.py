import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load normalized numeric data
data = pd.read_csv('data_normalized.csv')  # Your normalized CSV path

# Define features and target
y = data['tmode']
X = data.drop(columns=['tmode'])

# Pipeline for numeric data: impute missing values (if any) and scale data
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())  # Add scaling to normalize features
])

# Stratified 10-fold outer CV
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the MLPClassifier model (simple neural network)
mlp_model = MLPClassifier(random_state=42)

# Pipeline with the imputer, scaler, and MLPClassifier model
pipeline = Pipeline([
    ('imputer', numeric_transformer),
    ('mlp_classifier', mlp_model)
])

# Hyperparameter grid for the neural network (MLP)
param_grid = {
    'mlp_classifier__hidden_layer_sizes': [(100,), (200,), (100, 100)],  # Neural network hidden layers
    'mlp_classifier__activation': ['relu', 'tanh'],  # Activation function for MLP
    'mlp_classifier__solver': ['adam', 'sgd'],  # Optimizer for MLP
    'mlp_classifier__alpha': [0.0001, 0.001],  # Regularization parameter for MLP
    'mlp_classifier__max_iter': [1000, 2000],  # Increase the max iterations
    'mlp_classifier__early_stopping': [True],  # Enable early stopping
    'mlp_classifier__validation_fraction': [0.1],  # Fraction of training data to use for early stopping
    'mlp_classifier__n_iter_no_change': [10],  # Number of iterations with no improvement before stopping
}

# Metrics storage per model
accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrix_sum = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)  # Fixed line
roc_aucs = []
all_probabilities = []
all_true_labels = []
all_pred_labels = []

# Cross-validation loop
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    logger.info(f"Neural Network (MLP) - Fold {fold} processing...")

    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_test_bin = label_binarize(y_test, classes=np.unique(y))

    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=pipeline,
       param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f" Best params: {grid_search.best_params_}")

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
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
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

logger.info(f"Neural Network (MLP) - Cross-validation completed.\n")

# Average metrics reporting
logger.info(f"Neural Network (MLP) - Average Results Over 10 Folds:")
logger.info(f" Accuracy: {np.mean(accuracies):.4f}")
logger.info(f" Precision: {np.mean(precisions):.4f}")
logger.info(f" Recall: {np.mean(recalls):.4f}")
logger.info(f" F1-score: {np.mean(f1s):.4f}")
logger.info(f" ROC AUC (macro): {np.nanmean(roc_aucs):.4f}")
logger.info(f"Neural Network (MLP) - Aggregated Confusion Matrix:")
logger.info(conf_matrix_sum)

# Save aggregated confusion matrix to CSV
conf_matrix_df = pd.DataFrame(conf_matrix_sum, index=np.unique(y), columns=np.unique(y))
conf_matrix_df.to_csv(f'confusion_matrix_NN.csv')
logger.info(f"Neural Network (MLP) - Aggregated confusion matrix saved to 'confusion_matrix_NN.csv'")

# Save average metrics to a text file
with open(f'average_metrics_NN.txt', 'w') as f:
    f.write(f"Average Results Over 10 Folds for Neural Network (MLP):\n")
    f.write(f"Accuracy: {np.mean(accuracies):.4f}\n")
    f.write(f"Precision: {np.mean(precisions):.4f}\n")
    f.write(f"Recall: {np.mean(recalls):.4f}\n")
    f.write(f"F1-score: {np.mean(f1s):.4f}\n")
    f.write(f"ROC AUC (macro): {np.nanmean(roc_aucs):.4f}\n")

logger.info(f"Neural Network (MLP) - Average metrics saved to 'average_metrics_NN.txt'")

# Concatenate all probabilities and labels into single arrays
all_probabilities_np = np.vstack(all_probabilities)
all_true_labels_np = np.concatenate(all_true_labels)
all_pred_labels_np = np.concatenate(all_pred_labels)

# Create a DataFrame with predicted probabilities + true/predicted labels
prob_df = pd.DataFrame(all_probabilities_np, columns=[f'prob_class_{c}' for c in np.unique(y)])
prob_df['true_label'] = all_true_labels_np
prob_df['pred_label'] = all_pred_labels_np

logger.info(f"\nNeural Network (MLP) - Example of predicted probabilities DataFrame head:")
logger.info(prob_df.head())

# Save predicted probabilities and labels to CSV
prob_df.to_csv(f'predicted_probabilities_NN.csv', index=False)
logger.info(f"Neural Network (MLP) - Predicted probabilities saved to 'predicted_probabilities_NN.csv'")