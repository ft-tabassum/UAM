import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

data = pd.read_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/LightGBM_synthetic_population_predictions_weights.csv')

# Calculate rtts for UAM
# rtts = 1 - (UAM_TT / groundbased, motorized mode (i.e., car or PT) TT
data['rtts'] = 1 - (data['travel_time_Uam'] / data['autos_TT'])  # autos

# Calculate Weighted Travel Time Savings for UAM
# Weighted TT Savings for UAM = rtts * prob_AFT (probability of using UAM)
data['Weighted_TT_Savings'] = data['rtts'] * data['prob_mode_Autonomous Flying Taxi']



# Save the results to a new CSV file
data.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Output_analyze/weighted_tt_savings_results.csv', index=False)
