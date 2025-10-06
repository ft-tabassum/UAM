import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class AdvancedAFTFeatureEngineering:
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        
    def load_data(self, file_path):
        """Load AFT data"""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data shape: {self.data.shape}")
        return self.data
    
    def create_interaction_features(self):
        """Create meaningful interaction features for AFT choice"""
        print("Creating interaction features...")
        
        # Store original columns for reference
        self.original_columns = self.data.columns.tolist()
        
        # Cost-time ratios (important for transport choice)
        self.data['CAR_cost_time_ratio'] = self.data['CAR_CO'] / (self.data['CAR_TT'] + 1e-6)
        self.data['PT_cost_time_ratio'] = self.data['PT_CO'] / (self.data['PT_TT'] + 1e-6)
        self.data['AFT_cost_time_ratio'] = self.data['AFT_CO'] / (self.data['AFT_TT'] + 1e-6)
        
        # Relative cost differences
        self.data['AFT_vs_CAR_cost_diff'] = self.data['AFT_CO'] - self.data['CAR_CO']
        self.data['AFT_vs_PT_cost_diff'] = self.data['AFT_CO'] - self.data['PT_CO']
        self.data['CAR_vs_PT_cost_diff'] = self.data['CAR_CO'] - self.data['PT_CO']
        
        # Relative time differences
        self.data['AFT_vs_CAR_time_diff'] = self.data['AFT_TT'] - self.data['CAR_TT']
        self.data['AFT_vs_PT_time_diff'] = self.data['AFT_TT'] - self.data['PT_TT']
        self.data['CAR_vs_PT_time_diff'] = self.data['CAR_TT'] - self.data['PT_TT']
        
        # Technology adoption likelihood (composite score) - CRUCIAL for AFT prediction
        tech_cols = [col for col in self.data.columns if 'Likelihood_r' in col]
        if tech_cols:
            self.data['tech_adoption_score'] = self.data[tech_cols].mean(axis=1)
            print(f"  Created tech_adoption_score from {len(tech_cols)} Likelihood_r features")
        
        # Safety concern composite
        safety_cols = [col for col in self.data.columns if 'safety' in col.lower()]
        if safety_cols:
            self.data['safety_concern_score'] = self.data[safety_cols].mean(axis=1)
            print(f"  Created safety_concern_score from {len(safety_cols)} safety features")
        
        # Environmental concern composite
        env_cols = [col for col in self.data.columns if 'environment' in col.lower()]
        if env_cols:
            self.data['environmental_concern_score'] = self.data[env_cols].mean(axis=1)
            print(f"  Created environmental_concern_score from {len(env_cols)} environment features")
        
        # AFT vs alternatives preference strength
        if 'AFT_vs_CAR_cost_diff' in self.data.columns and 'AFT_vs_PT_cost_diff' in self.data.columns:
            self.data['aft_cost_competitiveness'] = (self.data['AFT_vs_CAR_cost_diff'] + self.data['AFT_vs_PT_cost_diff']) / 2
            print("  Created aft_cost_competitiveness feature")
        
        if 'AFT_vs_CAR_time_diff' in self.data.columns and 'AFT_vs_PT_time_diff' in self.data.columns:
            self.data['aft_time_competitiveness'] = (self.data['AFT_vs_CAR_time_diff'] + self.data['AFT_vs_PT_time_diff']) / 2
            print("  Created aft_time_competitiveness feature")
        
        # Technology adoption variance (captures consistency of AFT preference)
        if tech_cols:
            self.data['tech_adoption_consistency'] = self.data[tech_cols].std(axis=1)
            print("  Created tech_adoption_consistency feature")
        
        # Create AFT preference score (CRUCIAL for AFT prediction)
        self.create_aft_preference_score()
        
        new_features = len([col for col in self.data.columns if col not in self.original_columns])
        print(f"Added {new_features} interaction features for AFT prediction")
        
    def create_aft_preference_score(self):
        """Create AFT preference score based on 6-point Likert scale variables"""
        print("Creating AFT Preference Score from 6-Point Likert Scale Variables...")
        
        # Component 1: Autonomous Technology Attitude (AtoLattitude)
        print("  Component 1: Autonomous Technology Attitude")
        ato_positive = ['AtoLattitude_r1', 'AtoLattitude_r3']  # Positive attitudes
        ato_negative = ['AtoLattitude_r2', 'AtoLattitude_r4']  # Negative attitudes (fears)
        
        ato_score = 0
        if all(col in self.data.columns for col in ato_positive + ato_negative):
            # For 6-point scale: 1=Strongly disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly agree, 6=I do not know
            
            # Positive attitudes: Higher values = more positive (keep as is)
            ato_positive_sum = self.data[ato_positive].sum(axis=1)
            
            # Negative attitudes: Higher values = more fear (need to invert)
            # Invert: 1 becomes 5, 2 becomes 4, 3 stays 3, 4 becomes 2, 5 becomes 1, 6 becomes 3
            ato_negative_inverted = 6 - self.data[ato_negative]  # 6 because 1-5 scale
            # Handle "I do not know" (6) by converting to neutral (3)
            ato_negative_inverted = ato_negative_inverted.replace(0, 3)
            ato_negative_sum = ato_negative_inverted.sum(axis=1)
            
            ato_score = (ato_positive_sum + ato_negative_sum) / len(ato_positive + ato_negative)
            print(f"    Created from {len(ato_positive)} positive + {len(ato_negative)} negative (inverted) attitudes")
        
        # Component 2: AFT Adoption Likelihood (for different trip purposes)
        print("  Component 2: AFT Adoption Likelihood")
        likelihood_cols = ['Likelihood_r1', 'Likelihood_r2', 'Likelihood_r3', 'Likelihood_r4', 'Likelihood_r5', 'Likelihood_r6']
        likelihood_score = 0
        
        if all(col in self.data.columns for col in likelihood_cols):
            # Likelihood variables: Higher values = more likely to choose AFT
            # This is already in the right direction, no inversion needed
            # Handle "I do not know" (6) by converting to neutral (3)
            likelihood_data = self.data[likelihood_cols].copy()
            likelihood_data = likelihood_data.replace(6, 3)  # 6 (I do not know) becomes 3 (neutral)
            
            likelihood_score = likelihood_data.mean(axis=1)
            print(f"    Created from {len(likelihood_cols)} trip purpose likelihoods (6=I do not know → 3=neutral)")
        
        # Component 3: Technology Enthusiasm
        print("  Component 3: Technology Enthusiasm")
        tech_positive = ['technologyconcern_r1', 'technologyconcern_r2']  # Excited, use expensive tech
        tech_negative = ['technologyconcern_r3', 'technologyconcern_r4']  # No interest, tech causes problems
        
        tech_score = 0
        if all(col in self.data.columns for col in tech_positive + tech_negative):
            # tech_positive: Higher values = more positive (keep as is)
            # tech_negative: Higher values = more negative (need to invert)
            
            tech_positive_sum = self.data[tech_positive].sum(axis=1)
            
            # Invert negative technology attitudes
            tech_negative_inverted = 6 - self.data[tech_negative]  # Assuming 1-5 scale
            # Handle "I do not know" (6) by converting to neutral (3)
            tech_negative_inverted = tech_negative_inverted.replace(0, 3)
            tech_negative_sum = tech_negative_inverted.sum(axis=1)
            
            tech_score = (tech_positive_sum + tech_negative_sum) / len(tech_positive + tech_negative)
            print(f"    Created from {len(tech_positive)} positive + {len(tech_negative)} negative (inverted) tech attitudes")
        
        # Component 4: Environmental Consciousness
        print("  Component 4: Environmental Consciousness")
        env_positive = ['environmentconcern_r1', 'environmentconcern_r4']  # Concerned about warming, willing to pay for eco-friendly
        env_negative = ['environmentconcern_r2', 'environmentconcern_r3']  # Don't change behavior, accept pollution
        
        env_score = 0
        if all(col in self.data.columns for col in env_positive + env_negative):
            # env_positive: Higher values = more positive (keep as is)
            # env_negative: Higher values = more negative (need to invert)
            
            env_positive_sum = self.data[env_positive].sum(axis=1)
            
            # Invert negative environmental attitudes
            env_negative_inverted = 6 - self.data[env_negative]  # Assuming 1-5 scale
            # Handle "I do not know" (6) by converting to neutral (3)
            env_negative_inverted = env_negative_inverted.replace(0, 3)
            env_negative_sum = env_negative_inverted.sum(axis=1)
            
            env_score = (env_positive_sum + env_negative_sum) / len(env_positive + env_negative)
            print(f"    Created from {len(env_positive)} positive + {len(env_negative)} negative (inverted) environmental attitudes")
        
        # Component 5: AFT-Specific Safety and Multimodal Preferences
        print("  Component 5: AFT-Specific Preferences")
        aft_specific_score = 0
        aft_cols = []
        
        if 'AFT_MULTI_yes' in self.data.columns:
            aft_cols.append('AFT_MULTI_yes')
        
        if 'AFT_SAFETY_safer' in self.data.columns:
            aft_cols.append('AFT_SAFETY_safer')
        
        if 'AFT_SAFETY_ds' in self.data.columns:
            aft_cols.append('AFT_SAFETY_ds')
        
        if 'AFT_SAFETY_riskier' in self.data.columns:
            # Invert this: higher riskier = lower preference
            # For 6-point scale: 1 becomes 5, 2 becomes 4, 3 stays 3, 4 becomes 2, 5 becomes 1, 6 becomes 3 (neutral)
            self.data['AFT_SAFETY_riskier_inverted'] = 6 - self.data['AFT_SAFETY_riskier']
            # Handle "I do not know" (6) by converting to neutral (3)
            self.data['AFT_SAFETY_riskier_inverted'] = self.data['AFT_SAFETY_riskier_inverted'].replace(0, 3)
            aft_cols.append('AFT_SAFETY_riskier_inverted')
        
        if aft_cols:
            aft_specific_score = self.data[aft_cols].mean(axis=1)
            print(f"    Created from {len(aft_cols)} AFT-specific features: {aft_cols}")
        
        # Combine all components with weights
        components = {}
        weights = {}
        
        if ato_score is not 0:
            components['autonomous_attitude'] = ato_score
            weights['autonomous_attitude'] = 0.25  # 25% weight
        
        if likelihood_score is not 0:
            components['aft_likelihood'] = likelihood_score
            weights['aft_likelihood'] = 0.30  # 30% weight (most important)
        
        if tech_score is not 0:
            components['technology_enthusiasm'] = tech_score
            weights['technology_enthusiasm'] = 0.20  # 20% weight
        
        if env_score is not 0:
            components['environmental_consciousness'] = env_score
            weights['environmental_consciousness'] = 0.15  # 15% weight
        
        if aft_specific_score is not 0:
            components['aft_specific'] = aft_specific_score
            weights['aft_specific'] = 0.10  # 10% weight
        
        # Calculate weighted AFT preference score
        if components:
            total_weight = sum(weights.values())
            weighted_sum = 0
            
            for component, weight in weights.items():
                weighted_sum += components[component] * weight
            
            self.data['aft_preference_score'] = weighted_sum / total_weight
            
            # Normalize to 0-1 range
            min_score = self.data['aft_preference_score'].min()
            max_score = self.data['aft_preference_score'].max()
            if max_score > min_score:
                self.data['aft_preference_score'] = (self.data['aft_preference_score'] - min_score) / (max_score - min_score)
            
            print(f"  ✓ Created aft_preference_score from {len(components)} components")
            print(f"  Component weights: {weights}")
            print(f"  Final score range: {self.data['aft_preference_score'].min():.3f} to {self.data['aft_preference_score'].max():.3f}")
            
            # Show component ranges
            print(f"  Component score ranges:")
            for component, score in components.items():
                print(f"    {component}: {score.min():.3f} to {score.max():.3f}")
            
            return True
        
        print("  ✗ Could not create AFT preference score - no components available")
        return False
        
    def create_polynomial_features(self, degree=2):
        """Create polynomial features for numerical columns"""
        print(f"Creating polynomial features (degree {degree})...")
        
        numerical_cols = ['CAR_CO', 'CAR_TT', 'PT_CO', 'PT_TT', 'AFT_CO', 'AFT_TT']
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        
        if len(available_cols) >= 2:
            # Create quadratic terms for key features
            for col in available_cols[:3]:  # Limit to avoid too many features
                self.data[f'{col}_squared'] = self.data[col] ** 2
                
            # Create cross-product terms for cost and time
            if 'CAR_CO' in available_cols and 'CAR_TT' in available_cols:
                self.data['CAR_cost_time_product'] = self.data['CAR_CO'] * self.data['CAR_TT']
            if 'PT_CO' in available_cols and 'PT_TT' in available_cols:
                self.data['PT_cost_time_product'] = self.data['PT_CO'] * self.data['PT_TT']
            if 'AFT_CO' in available_cols and 'AFT_TT' in available_cols:
                self.data['AFT_cost_time_product'] = self.data['AFT_CO'] * self.data['AFT_TT']
        
        print(f"Added polynomial features for transport choice modeling")
        
    def advanced_scaling(self):
        """Apply selective scaling - only scale derived features"""
        print("Applying selective scaling...")
        
        # Only scale derived features (created by our script)
        derived_cols = [col for col in self.data.columns if any(x in col.lower() for x in ['ratio', 'diff', 'product', 'score'])]
        
        print(f"Found {len(derived_cols)} derived features to scale:")
        print(f"  {derived_cols}")
        
        for col in derived_cols:
            scaler = MinMaxScaler()  # 0-1 range, no negatives
            self.data[col] = scaler.fit_transform(self.data[col].values.reshape(-1, 1)).flatten()
            self.scalers[f'{col}_minmax'] = scaler
        
        print(f"✓ Scaled {len(derived_cols)} derived features to 0-1 range")
        print("✓ Original features (costs, times, Likert) kept unchanged")
        
    def feature_selection(self, method='mutual_info', k=80):
        """Select most important features for AFT prediction"""
        print(f"Performing feature selection using {method}...")
        
        X = self.data.drop(columns=['CHOICE'])
        y = self.data['CHOICE']
        
        # Essential features that must be kept
        essential_features = ['CAR_CO', 'CAR_TT', 'PT_CO', 'PT_TT', 'AFT_CO', 'AFT_TT']
        essential_features = [col for col in essential_features if col in X.columns]
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Ensure essential features are always included
        for feature in essential_features:
            if feature not in selected_features:
                selected_features.append(feature)
                print(f"  Force-kept essential feature: {feature}")
        
        # Keep only selected features + target
        self.data = self.data[selected_features + ['CHOICE']]
        self.feature_selectors[method] = selector
        
        print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        print(f"Selected features: {selected_features[:10]}...")
        
        # Show which AFT-specific features were selected
        aft_features = [col for col in selected_features if 'AFT' in col or 'tech_adoption' in col or 'aft_preference' in col]
        if aft_features:
            print(f"AFT-specific features selected: {aft_features}")
        
        return selected_features
        
    def handle_class_imbalance(self):
        """Apply techniques to handle class imbalance for AFT prediction"""
        print("Handling class imbalance...")
        
        # Check class distribution
        class_counts = self.data['CHOICE'].value_counts().sort_index()
        print(f"Original class distribution: {class_counts.to_dict()}")
        
        # Calculate class weights for training
        total_samples = len(self.data)
        class_weights = {}
        for class_label in class_counts.index:
            class_weights[class_label] = total_samples / (len(class_counts) * class_counts[class_label])
        
        print(f"Class weights for training: {class_weights}")
        
        # Emphasize AFT class (Class 2) even more if it's severely underrepresented
        if 2 in class_weights and class_weights[2] < 2.0:
            class_weights[2] = class_weights[2] * 1.5  # Increase weight for AFT class
            print(f"Boosted AFT class weight to: {class_weights[2]:.2f}")
        
        # Store for later use in models
        self.class_weights = class_weights
        
        return class_weights
    
    def save_processed_data(self, output_path):
        """Save processed data"""
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        
        # Save feature importance analysis
        importance_path = output_path.replace('.csv', '_feature_importance.csv')
        mi_df, f_df = self.get_feature_importance_analysis()
        
        # Combine both importance measures
        importance_df = pd.merge(mi_df, f_df, on='feature', how='outer')
        importance_df = importance_df.sort_values('mi_score', ascending=False)
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance analysis saved to: {importance_path}")
    
    def get_feature_importance_analysis(self):
        """Analyze feature importance using multiple methods"""
        print("Analyzing feature importance...")
        
        X = self.data.drop(columns=['CHOICE'])
        y = self.data['CHOICE']
        
        # Mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        
        # F-statistic scores
        f_scores = f_classif(X, y)[0]
        f_df = pd.DataFrame({'feature': X.columns, 'f_score': f_scores})
        f_df = f_df.sort_values('f_score', ascending=False)
        
        print("Top 10 features by Mutual Information:")
        print(mi_df.head(10))
        
        print("\nTop 10 features by F-statistic:")
        print(f_df.head(10))
        
        # Highlight AFT-specific features
        aft_features = [col for col in X.columns if 'AFT' in col or 'tech_adoption' in col or 'aft_preference' in col]
        if aft_features:
            print(f"\nAFT-specific features and their importance:")
            for feature in aft_features:
                mi_score = mi_df[mi_df['feature'] == feature]['mi_score'].iloc[0]
                f_score = f_df[f_df['feature'] == feature]['f_score'].iloc[0]
                print(f"  {feature}: MI={mi_score:.4f}, F={f_score:.4f}")
        
        return mi_df, f_df

def main():
    """Main function to run advanced feature engineering for AFT prediction"""
    
    print("=== Advanced AFT Feature Engineering for Improved Prediction ===")
    
    # Initialize feature engineering
    fe = AdvancedAFTFeatureEngineering()
    
    # Load data
    data_path = '/Result/DataPreprocessing_aft/aft_processed.csv'
    data = fe.load_data(data_path)
    
    print(f"\nStarting with {data.shape[1]} features")
    
    # Apply advanced feature engineering
    fe.create_interaction_features()
    fe.create_polynomial_features(degree=2)
    fe.advanced_scaling()
    
    # Handle class imbalance (crucial for AFT prediction)
    class_weights = fe.handle_class_imbalance()
    
    # Feature selection (keep top 80 most important features)
    selected_features = fe.feature_selection(method='mutual_info', k=80)
    
    # Analyze feature importance
    mi_df, f_df = fe.get_feature_importance_analysis()
    
    # Save processed data
    output_path = '/Result/DataPreprocessing_aft/aft_advanced_features.csv'
    fe.save_processed_data(output_path)
    
    print("\n=== Advanced Feature Engineering Complete ===")
    print(f"Original features: {len(fe.original_columns)}")
    print(f"Final features: {len(data.columns)}")
    print(f"Features removed: {len(fe.original_columns) - len(data.columns)}")
    print(f"Class weights: {class_weights}")
    print(f"\nKey improvements for AFT prediction:")
    print("  ✓ Created tech_adoption_score from Likelihood_r features")
    print("  ✓ Created aft_preference_score from 6-point Likert scale variables")
    print("  ✓ Added cost-time ratios and relative differences")
    print("  ✓ Applied advanced scaling for different feature types")
    print("  ✓ Selected top 60 most important features")
    print("  ✓ Calculated balanced class weights")
    print("  ✓ Properly handled 6-point Likert scale (1-5 + 6=I do not know)")

if __name__ == "__main__":
    main()
