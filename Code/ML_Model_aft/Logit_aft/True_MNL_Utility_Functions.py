"""
True MNL Model with Utility Functions
This script implements the actual MNL model with utility functions V1, V2, V3
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_utility_features(data):
    """Create features based on the exact utility functions from Biogeme"""
    
    print("Creating utility-based features...")
    
    # Scale time and cost variables by 10 (same as Biogeme)
    data['CAR_TT_SCALED'] = data['CAR_TT'] / 10
    data['CAR_COST_SCALED'] = data['CAR_CO'] / 10
    
    data['PT_TT_SCALED'] = data['PT_TT'] / 10
    data['PT_COST_SCALED'] = data['PT_CO'] / 10
    data['PT_INC_SCALED'] = data['PT_INC'] / 10
    
    data['AFT_TT_SCALED'] = data['AFT_TT'] / 10
    data['AFT_COST_SCALED'] = data['AFT_CO'] / 10
    
    print("Scaled variables created successfully")
    return data

def calculate_utility_functions(data, params):
    """Calculate utility functions V1, V2, V3 for each observation"""
    
    # Extract parameters
    # ASC parameters
    ASC_CAR = params[0]
    ASC_PT = params[1] 
    ASC_AFT = params[2]
    
    # Time and Cost parameters
    B_CAR_TIME = params[3]
    B_PT_TIME = params[4]
    B_AFT_TIME = params[5]
    B_CAR_COST = params[6]
    B_PT_COST = params[7]
    B_AFT_COST = params[8]
    B_PT_INC = params[9]
    
    # Safety parameters
    B_riskier_AFT = params[10]
    
    # Age parameters
    B_AGE3_CAR = params[11]
    B_AGE5_CAR = params[12]
    B_AGE6_MODES = params[13]
    B_OLDER_AUTO = params[14]
    B_AGE3_AFT = params[15]
    
    # Education parameters
    B_BSC_CAR = params[16]
    B_BSC_AFT = params[17]
    
    # Current Transport Mode parameters
    B_CARUSER_CAR = params[18]
    B_PTUSER_CAR = params[19]
    B_SMUSER_CAR = params[20]
    B_PTUSER_AUTO = params[21]
    B_CARUSER_AFT = params[22]
    B_SMUSER_AFT = params[23]
    
    # Car Availability parameters
    B_HAVECAR_CAR = params[24]
    B_NOCAR_CAR = params[25]
    B_HAVECAR_AFT = params[26]
    B_NOCAR_AFT = params[27]
    
    # Employment parameters
    B_WORKING_CAR = params[28]
    B_STUDENT_CAR = params[29]
    B_OTHERS_CAR = params[30]
    B_WORKING_AFT = params[31]
    B_STUDENT_AFT = params[32]
    
    # Trip Purpose parameters
    B_COM_CAR = params[33]
    B_NONCOM_CAR = params[34]
    B_COM_AFT = params[35]
    B_NONCOM_AFT = params[36]
    
    # Income parameters
    B_INCOME1_CAR = params[37]
    B_INCOME5_CAR = params[38]
    B_INCOME9_CAR = params[39]
    B_INCOME2_AFT = params[40]
    B_INCOME3_AFT = params[41]
    B_INCOME4_AFT = params[42]
    B_INCOME5_AFT = params[43]
    B_INCOME8_AFT = params[44]
    B_INCOME9_AFT = params[45]
    
    # Children parameters
    B_NOCHILDREN_AFT = params[46]
    
    # Calculate utility functions
    # V1 = B_CAR_TIME * CAR_TT_SCALED + B_CAR_COST * CAR_COST_SCALED + B_AGE3_CAR * AGE3 + B_AGE5_CAR * AGE5 + B_AGE6_MODES * AGE6 + B_BSC_CAR * BSC + B_CARUSER_CAR * CARUSER + B_PTUSER_CAR * PTUSER + B_SMUSER_CAR * SMUSER + B_HAVECAR_CAR * HAVECAR + B_NOCAR_CAR * NOCAR + B_WORKING_CAR * WORKING + B_STUDENT_CAR * STUDENT + B_OTHERS_CAR * OTHERS + B_COM_CAR * COM + B_NONCOM_CAR * NONCOM + B_INCOME1_CAR * INCOME1 + B_INCOME5_CAR * INCOME5 + B_INCOME9_CAR * INCOME9
    V1 = (ASC_CAR + 
          B_CAR_TIME * data['CAR_TT_SCALED'] + 
          B_CAR_COST * data['CAR_COST_SCALED'] + 
          B_AGE3_CAR * data['AGE3'] + 
          B_AGE5_CAR * data['AGE5'] + 
          B_AGE6_MODES * data['AGE6'] + 
          B_BSC_CAR * data['BSC'] + 
          B_CARUSER_CAR * data['CARUSER'] + 
          B_PTUSER_CAR * data['PTUSER'] + 
          B_SMUSER_CAR * data['SMUSER'] + 
          B_HAVECAR_CAR * data['HAVECAR'] + 
          B_NOCAR_CAR * data['NOCAR'] + 
          B_WORKING_CAR * data['WORKING'] + 
          B_STUDENT_CAR * data['STUDENT'] + 
          B_OTHERS_CAR * data['OTHERS'] + 
          B_COM_CAR * data['COM'] + 
          B_NONCOM_CAR * data['NONCOM'] + 
          B_INCOME1_CAR * data['INCOME1'] + 
          B_INCOME5_CAR * data['INCOME5'] + 
          B_INCOME9_CAR * data['INCOME9'])
    
    # VV2 = ASC_PT + B_PT_TIME * PT_TT_SCALED + B_PT_COST * PT_COST_SCALED + B_PT_INC * PT_INC_SCALED
    V2 = (ASC_PT + 
          B_PT_TIME * data['PT_TT_SCALED'] + 
          B_PT_COST * data['PT_COST_SCALED'] + 
          B_PT_INC * data['PT_INC_SCALED'])
    
    # V3 = ASC_AFT + B_AFT_TIME * AFT_TT_SCALED + B_AFT_COST * AFT_COST_SCALED + B_riskier_AFT * AFT_SAFETY_riskier + B_AGE3_AFT * AGE3 + B_OLDER_AUTO * AGE4 + B_OLDER_AUTO * AGE5 + B_AGE6_MODES * AGE6 + B_BSC_AFT * BSC + B_CARUSER_AFT * CARUSER + B_PTUSER_AUTO * PTUSER + B_SMUSER_AFT * SMUSER + B_HAVECAR_AFT * HAVECAR + B_NOCAR_AFT * NOCAR + B_WORKING_AFT * WORKING + B_STUDENT_AFT * STUDENT + B_COM_AFT * COM + B_NONCOM_AFT * NONCOM + B_INCOME2_AFT * INCOME2 + B_INCOME3_AFT * INCOME3 + B_INCOME4_AFT * INCOME4 + B_INCOME5_AFT * INCOME5 + B_INCOME8_AFT * INCOME8 + B_INCOME9_AFT * INCOME9 + B_NOCHILDREN_AFT * NOCHILDREN
    V3 = (ASC_AFT + 
          B_AFT_TIME * data['AFT_TT_SCALED'] + 
          B_AFT_COST * data['AFT_COST_SCALED'] + 
          B_riskier_AFT * data['AFT_SAFETY_riskier'] + 
          B_AGE3_AFT * data['AGE3'] + 
          B_OLDER_AUTO * data['AGE4'] + 
          B_OLDER_AUTO * data['AGE5'] + 
          B_AGE6_MODES * data['AGE6'] + 
          B_BSC_AFT * data['BSC'] + 
          B_CARUSER_AFT * data['CARUSER'] + 
          B_PTUSER_AUTO * data['PTUSER'] + 
          B_SMUSER_AFT * data['SMUSER'] + 
          B_HAVECAR_AFT * data['HAVECAR'] + 
          B_NOCAR_AFT * data['NOCAR'] + 
          B_WORKING_AFT * data['WORKING'] + 
          B_STUDENT_AFT * data['STUDENT'] + 
          B_COM_AFT * data['COM'] + 
          B_NONCOM_AFT * data['NONCOM'] + 
          B_INCOME2_AFT * data['INCOME2'] + 
          B_INCOME3_AFT * data['INCOME3'] + 
          B_INCOME4_AFT * data['INCOME4'] + 
          B_INCOME5_AFT * data['INCOME5'] + 
          B_INCOME8_AFT * data['INCOME8'] + 
          B_INCOME9_AFT * data['INCOME9'] + 
          B_NOCHILDREN_AFT * data['NOCHILDREN'])
    
    return V1, V2, V3

def calculate_probabilities(V1, V2, V3):
    """Calculate choice probabilities using MNL formula"""
    
    # Calculate exp(V) for each alternative
    exp_V1 = np.exp(V1)
    exp_V2 = np.exp(V2)
    exp_V3 = np.exp(V3)
    
    # Calculate denominator (sum of all exp(V))
    denominator = exp_V1 + exp_V2 + exp_V3
    
    # Calculate probabilities
    P1 = exp_V1 / denominator  # Probability of choosing Car
    P2 = exp_V2 / denominator  # Probability of choosing PT
    P3 = exp_V3 / denominator  # Probability of choosing AFT
    
    return P1, P2, P3

def log_likelihood(params, data, choices):
    """Calculate log-likelihood for parameter estimation"""
    
    try:
        # Calculate utility functions
        V1, V2, V3 = calculate_utility_functions(data, params)
        
        # Calculate probabilities
        P1, P2, P3 = calculate_probabilities(V1, V2, V3)
        
        # Calculate log-likelihood
        log_lik = 0
        for i, choice in enumerate(choices):
            if choice == 1:  # Car
                log_lik += np.log(P1.iloc[i] + 1e-10)  # Add small value to avoid log(0)
            elif choice == 2:  # PT
                log_lik += np.log(P2.iloc[i] + 1e-10)
            elif choice == 3:  # AFT
                log_lik += np.log(P3.iloc[i] + 1e-10)
        
        return -log_lik  # Minimize negative log-likelihood
        
    except:
        return 1e10  # Return large value if calculation fails

def estimate_mnl_parameters(data, choices):
    """Estimate MNL parameters using maximum likelihood"""
    
    print("Estimating MNL parameters...")
    
    # Initial parameter values (47 parameters total)
    initial_params = np.zeros(47)
    
    # Set some reasonable initial values
    initial_params[0] = 0.0   # ASC_CAR
    initial_params[1] = 0.0   # ASC_PT  
    initial_params[2] = 0.0   # ASC_AFT
    initial_params[3] = -0.1  # B_CAR_TIME (negative expected)
    initial_params[4] = -0.1  # B_PT_TIME (negative expected)
    initial_params[5] = -0.1  # B_AFT_TIME (negative expected)
    initial_params[6] = -0.1  # B_CAR_COST (negative expected)
    initial_params[7] = -0.1  # B_PT_COST (negative expected)
    initial_params[8] = -0.1  # B_AFT_COST (negative expected)
    initial_params[9] = -0.1  # B_PT_INC (negative expected)
    
    # Bounds for parameters
    bounds = [(-10, 10)] * 47  # All parameters between -10 and 10
    
    # Estimate parameters
    result = minimize(log_likelihood, initial_params, 
                     args=(data, choices), 
                     method='L-BFGS-B', 
                     bounds=bounds,
                     options={'maxiter': 1000, 'disp': True})
    
    if result.success:
        print("Parameter estimation successful!")
        print(f"Log-likelihood: {-result.fun:.4f}")
        return result.x
    else:
        print("Parameter estimation failed!")
        return None

def run_true_mnl_model():
    """Run true MNL model with utility functions"""
    
    print("="*70)
    print("RUNNING TRUE MNL MODEL WITH UTILITY FUNCTIONS")
    print("="*70)
    
    # Load data
    data_file = Path("D:/Thesis/UAM/Result/ML_models_aft/Logit_Models/biogeme_aft_data.csv")
    print(f"Loading data from: {data_file}")
    
    data = pd.read_csv(data_file)
    print(f"Data shape: {data.shape}")
    print(f"CHOICE distribution:")
    print(data['CHOICE'].value_counts().sort_index())
    
    # Create utility-based features
    data = create_utility_features(data)
    
    # Split data
    train_data, test_data, train_choices, test_choices = train_test_split(
        data, data['CHOICE'], test_size=0.2, random_state=42, stratify=data['CHOICE']
    )
    
    print(f"\nTrain set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Estimate parameters on training data
    params = estimate_mnl_parameters(train_data, train_choices)
    
    if params is not None:
        # Calculate utility functions and probabilities for test data
        V1_test, V2_test, V3_test = calculate_utility_functions(test_data, params)
        P1_test, P2_test, P3_test = calculate_probabilities(V1_test, V2_test, V3_test)
        
        # Make predictions
        probabilities = np.column_stack([P1_test, P2_test, P3_test])
        predictions = np.argmax(probabilities, axis=1) + 1  # Convert to 1,2,3
        
        # Calculate metrics
        accuracy = accuracy_score(test_choices, predictions)
        precision_macro = precision_score(test_choices, predictions, average='macro')
        recall_macro = recall_score(test_choices, predictions, average='macro')
        f1_macro = f1_score(test_choices, predictions, average='macro')
        precision_weighted = precision_score(test_choices, predictions, average='weighted')
        recall_weighted = recall_score(test_choices, predictions, average='weighted')
        f1_weighted = f1_score(test_choices, predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(test_choices, predictions, average=None)
        recall_per_class = recall_score(test_choices, predictions, average=None)
        f1_per_class = f1_score(test_choices, predictions, average=None)
        
        print(f"\nTEST SET RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"Precision (Weighted): {precision_weighted:.4f}")
        print(f"Recall (Weighted): {recall_weighted:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        print(f"\nPER-CLASS METRICS:")
        class_names = ['Car', 'Public Transport', 'Flying Taxi']
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  Precision: {precision_per_class[i]:.4f}")
            print(f"  Recall: {recall_per_class[i]:.4f}")
            print(f"  F1-Score: {f1_per_class[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(test_choices, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Car', 'Public Transport', 'Flying Taxi'],
                   yticklabels=['Car', 'Public Transport', 'Flying Taxi'])
        plt.title('True MNL Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save results
        output_dir = Path("D:/Thesis/UAM/Result/ML_models_aft/Logit_Models/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / 'true_mnl_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results
        with open(output_dir / 'true_mnl_results.txt', 'w') as f:
            f.write("TRUE MNL MODEL WITH UTILITY FUNCTIONS\n")
            f.write("="*50 + "\n\n")
            f.write("TEST SET METRICS:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision (Macro): {precision_macro:.4f}\n")
            f.write(f"Recall (Macro): {recall_macro:.4f}\n")
            f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
            f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
            f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
            f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n\n")
            f.write("PER-CLASS METRICS:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {precision_per_class[i]:.4f}\n")
                f.write(f"  Recall: {recall_per_class[i]:.4f}\n")
                f.write(f"  F1-Score: {f1_per_class[i]:.4f}\n")
        
        # Save predictions with probabilities
        predictions_df = pd.DataFrame({
            'True_Choice': test_choices,
            'Predicted_Choice': predictions,
            'Correct': (test_choices == predictions),
            'Prob_Car': P1_test,
            'Prob_PT': P2_test,
            'Prob_AFT': P3_test
        })
        predictions_df.to_csv(output_dir / 'true_mnl_predictions.csv', index=False)
        
        print(f"\nResults saved to: {output_dir}")
        print("True MNL model completed successfully!")
        
        return params, accuracy
        
    else:
        print("Model estimation failed!")
        return None, None

if __name__ == "__main__":
    params, accuracy = run_true_mnl_model()
