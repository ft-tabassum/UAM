import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
import warnings
import time
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()



class CustomDataset(Dataset):
    """Simple dataset class for PyTorch."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNeuralNetwork(nn.Module, BaseEstimator, ClassifierMixin):
    """Simple neural network with scikit-learn compatibility."""
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout_rate=0.4, 
                 learning_rate=0.001, batch_size=32, weight_decay=0.001):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 3))  # 3 classes: Car(0), Public Transport(1), UAM(2)
        self.network = nn.Sequential(*layers)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X, y):
        """Train the model."""
        # Create dataset and dataloader
        dataset = CustomDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Training loop with early stopping
        self.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 5  # Restored to reasonable patience
        
        for epoch in range(25):  # Increased to 25 epochs for better training
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        self.eval()
        dataset = CustomDataset(X, np.zeros(len(X)))  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        self.eval()
        dataset = CustomDataset(X, np.zeros(len(X)))  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        probabilities = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)

# Load data of UAM survey data
data = pd.read_csv("/Result/DataPreprocessing_aft/aft_normalized.csv")

# Define features and target
X = data.drop(columns=['CHOICE'])
y = data['CHOICE']

# Original classes
classes = np.unique(y)
n_classes = len(classes)

# Hyperparameter grid - full version for best accuracy
param_grid = {
    'hidden_sizes': [[128, 64], [256, 128]],  # All options for best performance
    'learning_rate': [0.0005, 0.001],                     # All learning rates
    'dropout_rate': [0.3, 0.4],                           # All dropout rates
    'batch_size': [32, 64],                               # Restored to original batch sizes
    'weight_decay': [0.0005, 0.001]                       # All weight decay values
}

# Split data into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# Setup cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Initialize storage for metrics
fold_metrics = {
    'accuracies': [], 'precisions': [], 'recalls': [],
    'f1s': [], 'roc_aucs': [], 'confusion_matrices': [], 
    'probabilities': [], 'true_labels': [], 'pred_labels': [],
    'best_params': [], 'train_accuracies': [], 'class_accuracies': []
}

# Initialize a list to store the probabilities
all_fold_probs = []

# Perform cross-validation with GridSearchCV in each fold
logger.info("Performing 10-fold cross-validation with GridSearchCV in each fold...")
logger.info("IMPROVED ACCURACY: Full parameter grid + 5-fold inner CV for better hyperparameter tuning")
start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val), 1):
    fold_start_time = time.time()
    logger.info(f"Processing Fold {fold}/10...")
    
    # Split data for this fold
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    # Create base model
    base_model = SimpleNeuralNetwork(X_train.shape[1])
    
    # Use GridSearchCV for hyperparameter tuning on training data
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,  #  hyperparameter tuning
        scoring='accuracy',
        n_jobs=1,  # Set to 1 to avoid multiprocessing issues
        verbose=0  # Reduced verbosity for cleaner output
    )
    
    # Fit GridSearchCV on training data
    grid_search.fit(X_train.values, y_train.values)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate training accuracy
    train_pred = best_model.predict(X_train.values)
    train_acc = accuracy_score(y_train, train_pred)
    fold_metrics['train_accuracies'].append(train_acc)
    
    # Make predictions on validation set
    val_pred = best_model.predict(X_val.values)
    val_proba = best_model.predict_proba(X_val.values)
    
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
    
    # Show timing information
    fold_time = time.time() - fold_start_time
    elapsed_time = time.time() - start_time
    avg_time_per_fold = elapsed_time / fold
    remaining_folds = 10 - fold
    estimated_remaining_time = remaining_folds * avg_time_per_fold
    
    logger.info(f"Fold {fold} completed in {fold_time:.1f}s")
    logger.info(f"Average time per fold: {avg_time_per_fold:.1f}s")
    logger.info(f"Estimated time remaining: {estimated_remaining_time:.1f}s ({remaining_folds} folds left)")
    logger.info("-" * 50)

# After all folds are processed, concatenate all fold probabilities into a single DataFrame
all_fold_probs_df = pd.concat(all_fold_probs, ignore_index=True)

# Save the aggregated probabilities to a single CSV file
all_fold_probs_df.to_csv('D:/Thesis/UAM/Result/ML_models_aft/Probabilities/Training_Probabilities/all_folds_probabilities_NeuralNetwork.csv', index=False)

total_time = time.time() - start_time
logger.info(f"Cross-validation completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
logger.info("All fold probabilities have been saved to 'D:/Thesis/UAM/Result/ML_models_aft/Probabilities/Training_Probabilities/all_folds_probabilities_NeuralNetwork.csv'.")

# Analyze parameter stability
param_counts = Counter()
for p in fold_metrics['best_params']:
    # Convert hidden_sizes list to tuple for hashing
    param_tuple = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()))
    param_counts[param_tuple] += 1

most_common_params = param_counts.most_common()
logger.info("\nParameter Stability Analysis:")
for params, count in most_common_params:
    logger.info(f"Parameters: {dict(params)}")
    logger.info(f"Selected in {count} out of 10 folds")

# Find most common best parameters across folds
# Convert parameters to hashable format for counting
param_list = []
for p in fold_metrics['best_params']:
    param_tuple = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()))
    param_list.append(param_tuple)

most_common_param_tuple = max(param_list, key=param_list.count)
# Convert back to dictionary format
best_params = dict(most_common_param_tuple)
# Convert hidden_sizes back to list if needed
if 'hidden_sizes' in best_params:
    best_params['hidden_sizes'] = list(best_params['hidden_sizes'])

logger.info(f"\nMost common best parameters across folds: {best_params}")

# Train final model on all training+validation data using most common best parameters
final_model = SimpleNeuralNetwork(X_train_val.shape[1], **best_params)
final_model.fit(X_train_val.values, y_train_val.values)

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

# Calculate mode-specific prediction error on test set
test_class_error = {}
for cls in classes:
    mask = y_test == cls
    if np.any(mask):
        # Mode-specific prediction error = 1 - per-class accuracy
        test_class_error[cls] = 1 - accuracy_score(y_test[mask], test_pred[mask])
    else:
        test_class_error[cls] = np.nan

# Save test set probabilities
test_probs_df = pd.DataFrame(test_proba, columns=classes)
test_probs_df.to_csv('D:/Thesis/UAM/Result/ML_models_aft/Probabilities/Testing_Probabilities/test_set_probabilities_NeuralNetwork.csv', index=False)

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
with open('/Result/ML_models_aft/Prediction_EvaluationMetrics/Result_NeuralNetwork.txt', 'w') as f:
    f.write("Results for Neural Network with 10-fold Cross-Validation:\n\n")
    
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
    
    f.write("\nFinal ML_models_aft Performance:\n")
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
    
    f.write("\nTest Set Mode-specific Prediction Error:\n")
    for cls, error in test_class_error.items():
        if not np.isnan(error):
            f.write(f"Class {cls}: {error:.4f}\n")
    
    f.write("\nTest Set Confusion Matrix:\n")
    f.write(f"{test_cm}\n")

# Save confusion matrix
conf_matrix_df = pd.DataFrame(test_cm, index=classes, columns=classes)
conf_matrix_df.to_csv('D:/Thesis/UAM/Result/ML_models_aft/Confusion_Matrix/CM_NeuralNetwork.csv', index=True)

logger.info("Cross-validation completed. Results saved to Result_NeuralNetwork.txt") 