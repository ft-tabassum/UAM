import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define your class names exactly as in your target variable
classes = ['Car', 'Public Transport', 'Car-Sharing', 'Ride-hailing', 'UAM']

# Load confusion matrix CSV (values only)
conf_matrix_df = pd.read_csv('confusion_matrix_XGBoost.csv', index_col=0)
conf_matrix_sum = conf_matrix_df.values

# Load feature importance CSV
feature_importance_df = pd.read_csv('feature_importances_XGBoost.csv')

# Load predicted probabilities  CSV
prob_df = pd.read_csv('predicted_probabilities_XGBoost.csv')

def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(8,6))
    plt.title("Confusion Matrix Heatmap - XGBoost")
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted Travel Mode')
    plt.ylabel('True Travel Mode')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance_df, top_n=15):
    top_features = feature_importance_df.head(top_n)
    plt.figure(figsize=(10,6))
    plt.title(f"Top {top_n} Feature Importances - XGBoost")
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_probability_distribution(prob_df, class_label):
    # For XGBoost, prob columns usually start at 0
    class_index = classes.index(class_label)  # no +1 offset
    prob_col = f'prob_class_{class_index}'
    df = prob_df.copy()
    df['correct'] = df['true_label'] == df['pred_label']

    plt.figure(figsize=(10,6))
    sns.kdeplot(df[df['correct']][prob_col], label='Correct Predictions', shade=True)
    sns.kdeplot(df[~df['correct']][prob_col], label='Wrong Predictions', shade=True)
    plt.title(f"Predicted Probability Distribution for Class {class_label} - XGBoost")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----- Run the visualizations for XGBoost -----
plot_confusion_matrix(conf_matrix_sum, classes)
plot_feature_importance(feature_importance_df, top_n=15)

# Change class_label to visualize other classes' distributions
plot_probability_distribution(prob_df, class_label=classes[0])
