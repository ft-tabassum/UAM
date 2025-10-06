import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import warnings
warnings.filterwarnings('ignore')

def create_biogeme_style_utility_variables(data):
    """
    Create utility variables based on the Biogeme model structure
    """
    
    # Scale variables (as done in Biogeme model)
    data['CAR_TT_SCALED'] = data['CAR_TT'] / 10
    data['CAR_COST_SCALED'] = data['CAR_CO'] / 10
    data['CAR_INC_SCALED'] = data['CAR_INC'] / 10
    
    data['PT_TT_SCALED'] = data['PT_TT'] / 10
    data['PT_COST_SCALED'] = data['PT_CO'] / 10
    data['PT_INC_SCALED'] = data['PT_INC'] / 10
    
    data['AFT_TT_SCALED'] = data['AFT_TT'] / 10
    data['AFT_COST_SCALED'] = data['AFT_CO'] / 10
    data['AFT_INC_SCALED'] = data['AFT_INC'] / 10
    
    # Alternative-specific variables for each mode
    car_vars = [
        'CAR_TT_SCALED', 'CAR_COST_SCALED', 'CAR_INC_SCALED'
    ]
    
    pt_vars = [
        'PT_TT_SCALED', 'PT_COST_SCALED', 'PT_INC_SCALED'
    ]
    
    aft_vars = [
        'AFT_TT_SCALED', 'AFT_COST_SCALED', 'AFT_INC_SCALED'
    ]
    
    # Safety variables (alternative-specific)
    safety_vars = [
        'AFT_SAFETY_riskier', 'AFT_SAFETY_safer', 'AFT_SAFETY_ds',
        'PT_SAFETY_safer', 'CAR_SAFETY_ds'
    ]
    
    # Multi-modal variables
    multimodal_vars = [
        'AFT_MULTI_yes', 'PT_MULTI_inpart', 'CAR_MULTI_no'
    ]
    
    # Demographic variables (generic - affect all alternatives)
    demographic_vars = [
        'female', 'age_18-25', 'age_26-35', 'age_36-45', 'age_46-55', 'age_56-65', 'age_65+',
        'employment_employed', 'employment_student', 'employment_others',
        'child_household_0', 'child_household_1', 'child_household_2', 'child_household_3andmore',
        'car_0', 'car_1', 'car_2', 'car_3andmore'
    ]
    
    # Attitude variables (generic)
    attitude_vars = [
        'Likelihood_r1', 'Likelihood_r2', 'Likelihood_r3', 'Likelihood_r4', 'Likelihood_r5', 'Likelihood_r6',
        'AtoLattitude_r1', 'AtoLattitude_r2', 'AtoLattitude_r3', 'AtoLattitude_r4',
        'technologyconcern_r1', 'technologyconcern_r2', 'technologyconcern_r3', 'technologyconcern_r4',
        'environmentconcern_r1', 'environmentconcern_r2', 'environmentconcern_r3', 'environmentconcern_r4',
        'satisfaction'
    ]
    
    # Current transport mode and other variables
    current_mode_vars = ['current_transportmode', 'driving_license_yes', 'driving_license_no']
    
    # Combine all variables
    all_vars = (car_vars + pt_vars + aft_vars + safety_vars + 
                multimodal_vars + demographic_vars + attitude_vars + current_mode_vars)
    
    # Check which variables exist in the data
    existing_vars = [var for var in all_vars if var in data.columns]
    
    print(f"Variables to use in Biogeme-style MNL: {len(existing_vars)}")
    print(f"Car-specific variables: {[v for v in car_vars if v in data.columns]}")
    print(f"PT-specific variables: {[v for v in pt_vars if v in data.columns]}")
    print(f"AFT-specific variables: {[v for v in aft_vars if v in data.columns]}")
    print(f"Safety variables: {[v for v in safety_vars if v in data.columns]}")
    print(f"Demographic variables: {len([v for v in demographic_vars if v in data.columns])}")
    
    return existing_vars, car_vars, pt_vars, aft_vars, safety_vars, demographic_vars, attitude_vars

def prepare_biogeme_mnl_data(data):
    """
    Prepare data in Biogeme style for MNL
    """
    
    # Get utility variables
    (existing_vars, car_vars, pt_vars, aft_vars, 
     safety_vars, demographic_vars, attitude_vars) = create_biogeme_style_utility_variables(data)
    
    # Prepare X and y
    X = data[existing_vars].copy()
    y = data['CHOICE'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove constant columns
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)
    
    # Scale features (important for MNL convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler, existing_vars

def run_biogeme_style_mnl():
    """Run MNL model based on Biogeme utility function structure"""
    
    print("Loading data for Biogeme-style MNL model...")
    data = pd.read_csv('D:/Thesis/UAM/Result/DataPreprocessing_aft/aft_processed.csv')
    
    print(f"Data shape: {data.shape}")
    print(f"Class distribution: {data['CHOICE'].value_counts().sort_index()}")
    
    # Prepare data with Biogeme-style utility variables
    X, y, scaler, feature_names = prepare_biogeme_mnl_data(data)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names[:15]}...")  # Show first 15
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Fit MNL model with Biogeme-style specification
    print("Fitting Biogeme-style MNL model...")
    
    try:
        # Add constant term
        X_train_const = sm.add_constant(X_train)
        
        # Fit MNL model
        mnl_model = MNLogit(y_train, X_train_const)
        mnl_result = mnl_model.fit(disp=True, maxiter=1000)
        
        print("Biogeme-style MNL model fitted successfully!")
        print(f"Log-likelihood: {mnl_result.llf:.4f}")
        print(f"AIC: {mnl_result.aic:.4f}")
        print(f"BIC: {mnl_result.bic:.4f}")
        
        # Make predictions
        X_test_const = sm.add_constant(X_test)
        predictions = mnl_result.predict(X_test_const)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted_classes)
        precision = precision_score(y_test, predicted_classes, average='macro', zero_division=0)
        recall = recall_score(y_test, predicted_classes, average='macro', zero_division=0)
        f1 = f1_score(y_test, predicted_classes, average='macro', zero_division=0)
        
        # Calculate AUROC
        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_test, classes=[0, 1, 2])
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([y_bin, 1 - y_bin])
            auroc = roc_auc_score(y_bin, predictions, average='macro', multi_class='ovr')
        except:
            auroc = np.nan
        
        # Per-class accuracy
        per_class_accuracy = []
        classes = ['Car', 'PT', 'FT']
        for i in range(3):
            if i in y_test.values:
                class_mask = (y_test == i)
                if class_mask.sum() > 0:
                    class_acc = accuracy_score(y_test[class_mask], predicted_classes[class_mask])
                    per_class_accuracy.append(class_acc)
                else:
                    per_class_accuracy.append(0.0)
            else:
                per_class_accuracy.append(0.0)
        
        # Print results
        print("\n" + "="*60)
        print("BIOGEME-STYLE MNL MODEL RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUROC: {auroc:.4f}")
        
        print(f"\nPer-class Accuracy:")
        for i, class_name in enumerate(classes):
            print(f"Class {i} ({class_name}): {per_class_accuracy[i]:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, predicted_classes))
        
        # Show model summary (coefficients)
        print(f"\nModel Summary (first 15 coefficients):")
        try:
            summary_table = mnl_result.summary().tables[1]
            for i in range(min(16, len(summary_table.data))):  # Skip header, show first 15
                print(summary_table.data[i])
        except:
            print("Could not display coefficient summary")
        
        # Save results
        results = {
            'Model': 'MNL_Biogeme_Style',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'AUROC': auroc,
            'Car_Accuracy': per_class_accuracy[0],
            'PT_Accuracy': per_class_accuracy[1],
            'FT_Accuracy': per_class_accuracy[2],
            'LogLikelihood': mnl_result.llf,
            'AIC': mnl_result.aic,
            'BIC': mnl_result.bic
        }
        
        results_df = pd.DataFrame([results])
        results_df.to_csv('D:/Thesis/UAM/Result/ML_models_aft/MNL_Biogeme_Style_Results.csv', index=False)
        
        print(f"\nResults saved to: D:/Thesis/UAM/Result/ML_models_aft/MNL_Biogeme_Style_Results.csv")
        
        return results, mnl_result
        
    except Exception as e:
        print(f"Error fitting Biogeme-style MNL model: {str(e)}")
        print("This might be due to:")
        print("1. Convergence issues")
        print("2. Perfect separation")
        print("3. Insufficient data")
        print("4. Multicollinearity")
        return None, None

if __name__ == "__main__":
    results, model = run_biogeme_style_mnl()
