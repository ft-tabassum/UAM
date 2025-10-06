import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class EnhancedAFTModels:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading enhanced AFT data...")
        self.data = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.X = self.data.drop(columns=['CHOICE'])
        self.y = self.data['CHOICE']
        
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Class distribution: {self.y.value_counts().sort_index().to_dict()}")
        
        return self.X, self.y
    
    def get_class_weights(self):
        """Calculate balanced class weights"""
        classes = np.unique(self.y)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y)
        return dict(zip(classes, class_weights))
    
    def create_sampling_strategies(self):
        """Create different sampling strategies for class imbalance"""
        strategies = {}
        
        # SMOTE - Synthetic Minority Over-sampling Technique
        strategies['smote'] = SMOTE(random_state=42, k_neighbors=3)
        
        # ADASYN - Adaptive Synthetic Sampling
        strategies['adasyn'] = ADASYN(random_state=42, n_neighbors=3)
        
        # SMOTEENN - SMOTE + Edited Nearest Neighbors
        strategies['smoteenn'] = SMOTEENN(random_state=42)
        
        # SMOTETomek - SMOTE + Tomek Links
        strategies['smotetomek'] = SMOTETomek(random_state=42)
        
        # Random Under-sampling (for comparison)
        strategies['random_under'] = RandomUnderSampler(random_state=42)
        
        return strategies
    
    def enhanced_random_forest(self, use_sampling=True, sampling_method='smote'):
        """Enhanced Random Forest with class imbalance handling"""
        print(f"Training Enhanced Random Forest with {sampling_method}...")
        
        # Base classifier with optimized parameters
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        if use_sampling:
            # Create sampling strategy
            sampling_strategies = self.create_sampling_strategies()
            sampler = sampling_strategies.get(sampling_method, sampling_strategies['smote'])
            
            # Create pipeline with sampling
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_rf)
            ])
            
            # Cross-validation with sampling
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            # Train final model
            pipeline.fit(self.X, self.y)
            self.models['rf_enhanced'] = pipeline
            
        else:
            # Use class weights instead of sampling
            class_weights = self.get_class_weights()
            base_rf.set_params(class_weight=class_weights)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(base_rf, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            base_rf.fit(self.X, self.y)
            self.models['rf_enhanced'] = base_rf
        
        return self.models['rf_enhanced']
    
    def enhanced_xgboost(self, use_sampling=True, sampling_method='smote'):
        """Enhanced XGBoost with class imbalance handling"""
        print(f"Training Enhanced XGBoost with {sampling_method}...")
        
        # Base classifier with optimized parameters
        base_xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        if use_sampling:
            sampling_strategies = self.create_sampling_strategies()
            sampler = sampling_strategies.get(sampling_method, sampling_strategies['smote'])
            
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_xgb)
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            pipeline.fit(self.X, self.y)
            self.models['xgb_enhanced'] = pipeline
            
        else:
            # Use scale_pos_weight for class imbalance
            class_counts = self.y.value_counts().sort_index()
            scale_pos_weight = class_counts.max() / class_counts.min()
            base_xgb.set_params(scale_pos_weight=scale_pos_weight)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(base_xgb, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            base_xgb.fit(self.X, self.y)
            self.models['xgb_enhanced'] = base_xgb
        
        return self.models['xgb_enhanced']
    
    def enhanced_lightgbm(self, use_sampling=True, sampling_method='smote'):
        """Enhanced LightGBM with class imbalance handling"""
        print(f"Training Enhanced LightGBM with {sampling_method}...")
        
        # Base classifier with optimized parameters
        base_lgb = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        if use_sampling:
            sampling_strategies = self.create_sampling_strategies()
            sampler = sampling_strategies.get(sampling_method, sampling_strategies['smote'])
            
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_lgb)
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            pipeline.fit(self.X, self.y)
            self.models['lgb_enhanced'] = pipeline
            
        else:
            # Use class weights
            class_weights = self.get_class_weights()
            base_lgb.set_params(class_weight=class_weights)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(base_lgb, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            base_lgb.fit(self.X, self.y)
            self.models['lgb_enhanced'] = base_lgb
        
        return self.models['lgb_enhanced']
    
    def enhanced_neural_network(self, use_sampling=True, sampling_method='smote'):
        """Enhanced Neural Network with class imbalance handling"""
        print(f"Training Enhanced Neural Network with {sampling_method}...")
        
        # Base classifier with optimized architecture
        base_nn = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        if use_sampling:
            sampling_strategies = self.create_sampling_strategies()
            sampler = sampling_strategies.get(sampling_method, sampling_strategies['smote'])
            
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_nn)
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            pipeline.fit(self.X, self.y)
            self.models['nn_enhanced'] = pipeline
            
        else:
            # Use class weights
            class_weights = self.get_class_weights()
            base_nn.set_params(class_weight=class_weights)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(base_nn, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            base_nn.fit(self.X, self.y)
            self.models['nn_enhanced'] = base_nn
        
        return self.models['nn_enhanced']
    
    def enhanced_svm(self, use_sampling=True, sampling_method='smote'):
        """Enhanced SVM with class imbalance handling"""
        print(f"Training Enhanced SVM with {sampling_method}...")
        
        # Base classifier with optimized parameters
        base_svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        if use_sampling:
            sampling_strategies = self.create_sampling_strategies()
            sampler = sampling_strategies.get(sampling_method, sampling_strategies['smote'])
            
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_svm)
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            pipeline.fit(self.X, self.y)
            self.models['svm_enhanced'] = pipeline
            
        else:
            # Use class weights
            class_weights = self.get_class_weights()
            base_svm.set_params(class_weight=class_weights)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(base_svm, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            base_svm.fit(self.X, self.y)
            self.models['svm_enhanced'] = base_svm
        
        return self.models['svm_enhanced']
    
    def enhanced_stacking(self, use_sampling=True, sampling_method='smote'):
        """Enhanced Stacking Ensemble with class imbalance handling"""
        print(f"Training Enhanced Stacking Ensemble with {sampling_method}...")
        
        # Base estimators
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        if use_sampling:
            sampling_strategies = self.create_sampling_strategies()
            sampler = sampling_strategies.get(sampling_method, sampling_strategies['smote'])
            
            # Create sampling pipeline for each base estimator
            base_pipelines = []
            for name, estimator in base_estimators:
                pipeline = ImbPipeline([
                    ('sampler', sampler),
                    ('classifier', estimator)
                ])
                base_pipelines.append((name, pipeline))
            
            # Final stacking with sampling
            from sklearn.ensemble import StackingClassifier
            stacking = StackingClassifier(
                estimators=base_pipelines,
                final_estimator=meta_learner,
                cv=3,
                stack_method='predict_proba'
            )
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(stacking, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            stacking.fit(self.X, self.y)
            self.models['stacking_enhanced'] = stacking
            
        else:
            # Use class weights
            class_weights = self.get_class_weights()
            
            # Set class weights for base estimators
            for name, estimator in base_estimators:
                if hasattr(estimator, 'class_weight'):
                    estimator.set_params(class_weight=class_weights)
                elif hasattr(estimator, 'scale_pos_weight'):
                    class_counts = self.y.value_counts().sort_index()
                    scale_pos_weight = class_counts.max() / class_counts.min()
                    estimator.set_params(scale_pos_weight=scale_pos_weight)
            
            meta_learner.set_params(class_weight=class_weights)
            
            from sklearn.ensemble import StackingClassifier
            stacking = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=3,
                stack_method='predict_proba'
            )
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(stacking, self.X, self.y, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            stacking.fit(self.X, self.y)
            self.models['stacking_enhanced'] = stacking
        
        return self.models['stacking_enhanced']
    
    def evaluate_all_models(self, test_size=0.2):
        """Evaluate all enhanced models"""
        print("\n=== Evaluating All Enhanced Models_oldSurveydata ===")
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Train on training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Per-class accuracy
            per_class_accuracy = {}
            for i in range(len(conf_matrix)):
                per_class_accuracy[f'Class_{i}'] = conf_matrix[i][i] / conf_matrix[i].sum()
            
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'per_class_accuracy': per_class_accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Overall Accuracy: {accuracy:.4f}")
            print(f"Per-class Accuracy: {per_class_accuracy}")
            print(f"Confusion Matrix:\n{conf_matrix}")
        
        self.results = results
        return results
    
    def save_results(self, output_path):
        """Save evaluation results"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'accuracy': float(result['accuracy']),
                'per_class_accuracy': result['per_class_accuracy'],
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    """Main function to run enhanced model training"""
    
    # Initialize enhanced models
    data_path = '/Result/DataPreprocessing_aft/aft_advanced_features.csv'
    em = EnhancedAFTModels(data_path)
    
    # Load data
    X, y = em.load_data()
    
    print("\n=== Training Enhanced Models_oldSurveydata ===")
    
    # Train all enhanced models
    em.enhanced_random_forest(use_sampling=True, sampling_method='smote')
    em.enhanced_xgboost(use_sampling=True, sampling_method='smote')
    em.enhanced_lightgbm(use_sampling=True, sampling_method='smote')
    em.enhanced_neural_network(use_sampling=True, sampling_method='smote')
    em.enhanced_svm(use_sampling=True, sampling_method='smote')
    em.enhanced_stacking(use_sampling=True, sampling_method='smote')
    
    # Evaluate all models
    results = em.evaluate_all_models(test_size=0.2)
    
    # Save results
    output_path = '/Result/ML_models_aft/Enhanced_AFT_Results.json'
    em.save_results(output_path)
    
    print("\n=== Enhanced Model Training Complete ===")
    print("All models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
