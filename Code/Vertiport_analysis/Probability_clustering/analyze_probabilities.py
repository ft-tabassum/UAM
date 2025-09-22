import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

# Get UAM probabilities
uam_probs = df['prob_mode_Autonomous Flying Taxi']

print('UAM Probability Distribution Analysis:')
print('=' * 50)
print(f'Total trips: {len(uam_probs):,}')
print(f'Mean: {uam_probs.mean():.4f}')
print(f'Std: {uam_probs.std():.4f}')
print(f'Min: {uam_probs.min():.4f}')
print(f'Max: {uam_probs.max():.4f}')
print(f'Median: {uam_probs.median():.4f}')
print(f'25th percentile: {uam_probs.quantile(0.25):.4f}')
print(f'75th percentile: {uam_probs.quantile(0.75):.4f}')
print(f'Range: {uam_probs.max() - uam_probs.min():.4f}')
print(f'Coefficient of Variation: {uam_probs.std() / uam_probs.mean():.4f}')

print('\nProbability Distribution:')
print('-' * 30)
print(f'Trips with prob < 0.1: {np.sum(uam_probs < 0.1):,} ({np.sum(uam_probs < 0.1) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob < 0.2: {np.sum(uam_probs < 0.2):,} ({np.sum(uam_probs < 0.2) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob < 0.3: {np.sum(uam_probs < 0.3):,} ({np.sum(uam_probs < 0.3) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob < 0.4: {np.sum(uam_probs < 0.4):,} ({np.sum(uam_probs < 0.4) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob < 0.5: {np.sum(uam_probs < 0.5):,} ({np.sum(uam_probs < 0.5) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob >= 0.5: {np.sum(uam_probs >= 0.5):,} ({np.sum(uam_probs >= 0.5) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob >= 0.6: {np.sum(uam_probs >= 0.6):,} ({np.sum(uam_probs >= 0.6) / len(uam_probs) * 100:.1f}%)')
print(f'Trips with prob >= 0.7: {np.sum(uam_probs >= 0.7):,} ({np.sum(uam_probs >= 0.7) / len(uam_probs) * 100:.1f}%)')

print('\nWeight Distribution (for clustering):')
print('-' * 40)
# Show how weights would be distributed for clustering
weights = np.concatenate([uam_probs, uam_probs])  # Origins + destinations
print(f'Weight mean: {weights.mean():.4f}')
print(f'Weight std: {weights.std():.4f}')
print(f'Weight min: {weights.min():.4f}')
print(f'Weight max: {weights.max():.4f}')
print(f'Weight range: {weights.max() - weights.min():.4f}')

print('\nWeight Categories:')
print('-' * 20)
print(f'Very low weights (< 0.1): {np.sum(weights < 0.1):,} ({np.sum(weights < 0.1) / len(weights) * 100:.1f}%)')
print(f'Low weights (0.1-0.2): {np.sum((weights >= 0.1) & (weights < 0.2)):,} ({np.sum((weights >= 0.1) & (weights < 0.2)) / len(weights) * 100:.1f}%)')
print(f'Medium weights (0.2-0.3): {np.sum((weights >= 0.2) & (weights < 0.3)):,} ({np.sum((weights >= 0.2) & (weights < 0.3)) / len(weights) * 100:.1f}%)')
print(f'High weights (0.3-0.4): {np.sum((weights >= 0.3) & (weights < 0.4)):,} ({np.sum((weights >= 0.3) & (weights < 0.4)) / len(weights) * 100:.1f}%)')
print(f'Very high weights (>= 0.4): {np.sum(weights >= 0.4):,} ({np.sum(weights >= 0.4) / len(weights) * 100:.1f}%)')

print('\nConclusion:')
print('-' * 15)
if uam_probs.std() / uam_probs.mean() < 0.3:
    print('❌ LOW VARIATION: UAM probabilities have low variation')
    print('   → Weighted K-means behaves like unweighted K-means')
    print('   → Algorithm converges quickly because weights are similar')
elif np.sum(uam_probs < 0.1) / len(uam_probs) > 0.5:
    print('❌ MOSTLY LOW PROBABILITIES: Most trips have very low UAM probabilities')
    print('   → Most points have similar low weights')
    print('   → Algorithm doesn\'t have strong demand variation to optimize')
else:
    print('✅ GOOD VARIATION: UAM probabilities have sufficient variation')
    print('   → Weighted K-means should work effectively')
