import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Feature importance data for all three models
# Use np.nan for features not in a model's top 10 (so they won't be plotted)
data = {
    'Feature': [
        'Flying Taxi Travel cost',
        'Car Travel Time',
        'PT Travel Time',
        'Recreational Purpose',
        'Work Purpose',
        'Social Activity Purpose',
        'Current Transport Mode',
        'Child 3 or more',
        'Fear of Taking Flying Taxi',
        'Business Purpose',
        'No Car',
        'Employment',
        'Female',
        'Not Concerned about Environment',
        'Acceptable to Cause some Pollution',
        'Concerned about Global Warming',
        'Education Purpose'
    ],
    'XGBoost': [0.0235, 0.0350, 0.0268, np.nan, np.nan, 0.0186, 0.0361, 0.0589, np.nan, 0.0173, 
                0.0225, 0.0209, 0.0180, np.nan, np.nan, np.nan, np.nan],
    'LGBM': [0.0411, 0.0426, 0.0385, 0.0452, 0.0411, 0.0368, np.nan, np.nan, 0.0398, np.nan,
             np.nan, np.nan, np.nan, 0.0401, 0.0387, 0.0357, np.nan],
    'RF': [0.0720, 0.0695, 0.0542, 0.0419, 0.0355, 0.0301, 0.0257, np.nan, 0.0274, 0.0281,
           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0282]
}

df = pd.DataFrame(data)

# Create output directory path
output_dir = 'D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/graphs_visualize/'

print("=" * 80)
print("FEATURE IMPORTANCE COMPARISON: XGBoost vs LGBM vs Random Forest")
print("=" * 80)

# =============================================================================
# VISUALIZATION: VERTICAL GROUPED BAR CHART
# Comparing feature importance across XGBoost, LGBM, and Random Forest
# Vertical bars make it much clearer which bars belong to which feature
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 10))

x_pos = np.arange(len(df))
width = 0.25

# Create vertical bars with color fill - NaN values won't be plotted
# Bars are grouped together for each feature on x-axis
bars1 = ax.bar(x_pos - width, df['XGBoost'], width, 
               label='XGBoost', color='#4472C4', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x_pos, df['LGBM'], width, 
               label='LGBM', color='#ED7D31', alpha=0.8, edgecolor='black', linewidth=1)
bars3 = ax.bar(x_pos + width, df['RF'], width, 
               label='Random Forest', color='#70AD47', alpha=0.8, edgecolor='black', linewidth=1)

# Customize
ax.set_xlabel('Features', fontsize=18, fontweight='bold')
ax.set_ylabel('Importance', fontsize=18, fontweight='bold')

# Set x-axis ticks and labels 
ax.set_xticks(x_pos)
ax.set_xticklabels(df['Feature'], rotation=30, ha='right', fontsize=12)

ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=14, loc='upper right', framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(output_dir + 'feature_importance_grouped_bars.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved: feature_importance_grouped_bars.png")
print("  (Vertical bars - much clearer grouping by feature!)")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for model in ['XGBoost', 'LGBM', 'RF']:
    print(f"\n{model}:")
    model_data = df[model].dropna()  # Remove NaN values
    if len(model_data) > 0:
        top_feature_idx = df[model].idxmax()
        print(f"  Top feature: {df.loc[top_feature_idx, 'Feature']} ({df[model].max():.4f})")
        print(f"  Average importance: {model_data.mean():.4f}")
        print(f"  Total importance (top 10): {model_data.sum():.4f}")
        print(f"  Number of features in top 10: {len(model_data)}")

# Find common important features (appear in all models' top 10)
xgboost_features = set(df[df['XGBoost'].notna()]['Feature'])
lgbm_features = set(df[df['LGBM'].notna()]['Feature'])
rf_features = set(df[df['RF'].notna()]['Feature'])

common_features = xgboost_features & lgbm_features & rf_features

print(f"\n\nCommon Features (appear in all 3 models' top 10):")
if common_features:
    print(f"  Total: {len(common_features)} features")
    for feature in sorted(common_features):
        print(f"  - {feature}")
        row = df[df['Feature'] == feature].iloc[0]
        print(f"    XGBoost: {row['XGBoost']:.4f}, LGBM: {row['LGBM']:.4f}, RF: {row['RF']:.4f}")
else:
    print("  No features appear in all three models' top 10")

print("\n" + "=" * 80)
print("All visualizations completed successfully!")
print("=" * 80)

