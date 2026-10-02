[README.md](https://github.com/user-attachments/files/32967598/README.md)
# Demand-driven Vertiport Siting by Machine Learning for UAM Network Expansion

Code for my master's thesis at the Chair of Transportation Systems Engineering, Technical University of Munich (2025).

The project identifies vertiport locations for Urban Air Mobility (UAM) in the Munich metropolitan region. It combines machine learning models of travel mode choice with a demand-weighted clustering approach.

## Overview

The workflow has two parts:

1. **Mode choice modelling.** Six machine learning models are trained on stated preference survey data to predict the choice between car, public transport and UAM.
2. **Vertiport siting and evaluation.** The selected model (LightGBM) predicts UAM choice probabilities for trips in a synthetic population of the Munich region. These probabilities are used as weights in a demand-weighted K-means++ clustering to locate 74 vertiports. The resulting network is then evaluated in terms of demand coverage, travel time savings, mode share and congestion effects.

## Key results

- XGBoost achieved the highest overall accuracy (79.15%), while LightGBM achieved the highest UAM-specific accuracy (69.12%) and was therefore used for demand prediction.
- 74 vertiports covered 58.8% of the potential demand within a 5 km service area.
- The average travel time savings ratio was 11.08%, with larger savings for longer trips.
- The predicted UAM trip share was 3.28%.

## Repository structure

```
Code/
├── ML_Model_aft/                      # Part 1: mode choice modelling
│   ├── DataPreprocessing_aft/
│   │   ├── DataProcessing_aft.py      # Cleans survey data and encodes the choice variable
│   │   └── DataNormalization_aft.py   # Normalises numerical features
│   └── ML_models_aft/
│       ├── Random_Forest_aft.py
│       ├── Xgboost_aft.py
│       ├── LightGBM_aft.py
│       ├── SVM_aft.py
│       ├── Neural_Network_aft.py      # Feedforward neural network in PyTorch
│       ├── Stacking_aft.py            # Stacking ensemble
│       └── class_accuracy_heatmap_aft.py  # Class-wise accuracy comparison
│
└── Vertiport_analysis/                # Part 2: vertiport siting and evaluation
    ├── Lightermodel_aft/
    │   ├── Lighter_DataProcessing_aft.py  # Survey data restricted to features available in the synthetic population
    │   └── LightGBM_Model_Training.py     # Retrains LightGBM on these features
    ├── Synthetic_population/
    │   ├── microdata_trips.py         # Merges household, person and trip data 
    │   ├── microdata_trip_purpose.py  # Same as microdata_trips.py, but keeps the trip purpose
    │   ├── Mapping.py                 # Maps attributes (age, income, gender) to survey categories
    │   ├── DataPreprocessing_ML.py    # Prepares features for prediction
    │   └── tt_check.py                # Checks unusually long public transport travel times (over 100 min)
    ├── Probability_clustering/
    │   ├── analyze_probabilities.py   # Explores predicted UAM probabilities
    │   └── Weighted_clustering.py     # Demand-weighted K-means++ clustering
    └── Output_analyze/
        ├── coverage/                  # Demand coverage of the vertiport network
        ├── TT/                        # Travel time savings
        ├── modeShare/                 # Mode share and UAM replacement patterns
        └── congestion/                # Effect on road congestion
```

## Workflow

Run the scripts in this order:

1. `ML_Model_aft/DataPreprocessing_aft/`: data processing, then normalisation
2. `ML_Model_aft/ML_models_aft/`: train and evaluate the six models
3. `Vertiport_analysis/Lightermodel_aft/`: prepare the reduced feature set and retrain LightGBM
4. `Vertiport_analysis/Synthetic_population/`: prepare the synthetic population data
5. `Vertiport_analysis/Probability_clustering/Weighted_clustering.py`: predict UAM probabilities and locate vertiports
6. `Vertiport_analysis/Output_analyze/`: evaluate the vertiport network

## Methods

- **Models:** Random Forest, XGBoost, LightGBM, SVM, feedforward neural network and a stacking ensemble
- **Tuning and validation:** hyperparameter optimization with GridSearchCV inside stratified cross-validation
- **Evaluation:** accuracy, precision, recall, F1-score, AUROC and class-wise accuracy
- **Clustering:** demand-weighted K-means++ using predicted UAM choice probabilities as weights
- **Reproducibility:** fixed random seeds, scikit-learn pipelines and logging

## Main libraries

pandas, NumPy, SciPy, scikit-learn, LightGBM, XGBoost, matplotlib, seaborn, joblib

## Data

The data are not included in this repository:

- **Stated preference survey data** collected by Fu et al. (2019)
- **Synthetic population and trip data** from the SILO and MITO models (Moeckel et al., 2020)

Both datasets were provided for the thesis and cannot be shared publicly. The code therefore shows the full workflow but cannot be run without access to the data.

## Author

Fariha Tabassum
