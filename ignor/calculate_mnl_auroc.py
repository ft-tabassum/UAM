import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Load MNL predictions
mnl_predictions = pd.read_csv('../Result/ML_models_aft/Logit_Models/output/true_mnl_predictions.csv')

print("="*80)
print("CALCULATING AUROC FOR MNL MODEL")
print("="*80)

# Extract true labels and predicted probabilities
y_true = mnl_predictions['True_Choice'].values
y_proba = mnl_predictions[['Prob_Car', 'Prob_PT', 'Prob_AFT']].values

# Map choices to 0, 1, 2 (Car, PT, AFT)
# Based on the data, it looks like 1=Car, 2=PT, 3=AFT
choice_mapping = {1: 0, 2: 1, 3: 2}
y_true_mapped = np.array([choice_mapping[choice] for choice in y_true])

print(f"\nData Summary:")
print(f"Number of samples: {len(y_true)}")
print(f"Unique classes in data: {np.unique(y_true)}")
print(f"Mapped classes: {np.unique(y_true_mapped)}")
print(f"\nClass distribution:")
for cls in [1, 2, 3]:
    count = np.sum(y_true == cls)
    pct = 100 * count / len(y_true)
    cls_name = {1: 'Car', 2: 'PT', 3: 'AFT'}[cls]
    print(f"  {cls_name} (class {cls}): {count} samples ({pct:.2f}%)")

# Check probabilities sum to 1
prob_sums = y_proba.sum(axis=1)
print(f"\nProbability sums (should be ~1.0):")
print(f"  Min: {prob_sums.min():.6f}")
print(f"  Max: {prob_sums.max():.6f}")
print(f"  Mean: {prob_sums.mean():.6f}")

# Binarize true labels for multi-class ROC AUC
classes = [0, 1, 2]  # Car, PT, AFT
y_true_binary = label_binarize(y_true_mapped, classes=classes)

# Calculate AUROC
try:
    auroc_ovr = roc_auc_score(y_true_binary, y_proba, average='macro', multi_class='ovr')
    print(f"\n" + "="*80)
    print(f"MNL MODEL AUROC")
    print(f"="*80)
    print(f"\nAUROC (One-vs-Rest, Macro Average): {auroc_ovr:.4f}")
    
    # Calculate per-class AUROC
    print(f"\nPer-Class AUROC:")
    for i, cls_name in enumerate(['Private Car', 'Public Transport', 'Flying Taxi']):
        auroc_class = roc_auc_score(y_true_binary[:, i], y_proba[:, i])
        print(f"  {cls_name}: {auroc_class:.4f}")
    
    # Compare with ML models
    print(f"\n" + "="*80)
    print(f"COMPARISON WITH ML MODELS")
    print(f"="*80)
    
    ml_aurocs = {
        'LightGBM': 0.9134,
        'XGBoost': 0.9131,
        'Neural Network': 0.9089,
        'SVM': 0.9028,
        'Random Forest': 0.8987,
        'Stacking': 0.8904,
        'MNL': auroc_ovr
    }
    
    print(f"\nModel                AUROC     Rank")
    print(f"-" * 40)
    sorted_models = sorted(ml_aurocs.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, auroc) in enumerate(sorted_models, 1):
        print(f"{model:20s} {auroc:.4f}    {rank}")
    
    # Calculate gap
    best_ml = max([v for k, v in ml_aurocs.items() if k != 'MNL'])
    gap = best_ml - auroc_ovr
    print(f"\n" + "="*80)
    print(f"KEY FINDINGS:")
    print(f"="*80)
    print(f"MNL AUROC: {auroc_ovr:.4f}")
    print(f"Best ML AUROC: {best_ml:.4f} (LightGBM)")
    print(f"Performance Gap: {gap:.4f} ({100*gap:.2f} percentage points)")
    
    if auroc_ovr >= 0.80:
        interpretation = "good discrimination"
    elif auroc_ovr >= 0.70:
        interpretation = "fair discrimination"
    else:
        interpretation = "poor discrimination"
    
    print(f"\nInterpretation: MNL shows {interpretation}")
    print(f"All ML models achieved AUROC > 0.89 (excellent discrimination)")
    print(f"MNL AUROC is {gap:.4f} points lower than the best ML model")
    
    # Save results
    results_df = pd.DataFrame(sorted_models, columns=['Model', 'AUROC'])
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_df.to_csv('Result/ML_models_aft/Model_Comparison/all_models_auroc_with_mnl.csv', index=False)
    print(f"\n✅ Results saved to: all_models_auroc_with_mnl.csv")
    
except Exception as e:
    print(f"\n❌ Error calculating AUROC: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n" + "="*80)
print(f"ANALYSIS COMPLETE")
print(f"="*80)

