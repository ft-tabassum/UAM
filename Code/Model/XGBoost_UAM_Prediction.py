import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    print("\n=== Loading Data ===")
    # Training/validation data
    train_data = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/reduced_model_normalized.csv')
    print(f"Training/validation data shape: {train_data.shape}")
    # Testing data
    test_data = pd.read_csv('D:/PythonProject/Result/Data_Preprocessing/synthetic_data_normalized.csv')
    print(f"Testing data shape: {test_data.shape}")
    # Remove class 5 from synthetic data
    test_data = test_data[test_data['mode'] != 5]
    print(f"Testing data shape after removing class 5: {test_data.shape}")
    # Convert mode columns to int64 for consistency
    train_data['mode'] = train_data['mode'].astype('int64')
    test_data['mode'] = test_data['mode'].astype('int64')
    # Label encoding BEFORE splitting
    le = LabelEncoder()
    all_classes = pd.concat([train_data['mode'], test_data['mode']]).unique()
    le.fit(all_classes)
    print(f"Classes for encoding: {list(le.classes_)}")
    print("Label encoding mapping (encoded value -> original mode):")
    for idx, val in enumerate(le.classes_):
        print(f"  {idx} -> {val}")
    train_data['mode_enc'] = le.transform(train_data['mode'])
    test_data['mode_enc'] = le.transform(test_data['mode'])
    return train_data, test_data, le

def split_train_val(train_data):
    print("\n=== Splitting Training/Validation Data ===")
    X = train_data.drop(columns=['mode', 'mode_enc'])
    y = train_data['mode_enc']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def hyperparameter_tuning(X_train, y_train):
    print("\n=== Hyperparameter Tuning (10-fold CV) ===")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss'
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    return grid_search.best_params_, grid_search.best_score_

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    print("\n=== Training Final Model (No Early Stopping) ===")
    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_save(model, X_train, y_train, X_val, y_val, le, test_data):
    print("\n=== Evaluation and Results ===")
    # Validation
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=[str(c) for c in le.classes_]))
    # Training
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.4f}")
    # Test (synthetic)
    X_test = test_data.drop(columns=['mode', 'mode_enc'])
    y_test = test_data['mode_enc']
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Synthetic test accuracy: {test_acc:.4f}")
    print("Synthetic Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=[str(c) for c in le.classes_]))
    # UAM probabilities
    uam_idx = list(le.classes_).index(4)
    uam_probs = y_test_proba[:, uam_idx]
    print(f"UAM mean probability: {uam_probs.mean():.4f}")
    # Save results
    os.makedirs('D:/PythonProject/Result/Model', exist_ok=True)
    os.makedirs('D:/PythonProject/Result/Feature_Importance', exist_ok=True)
    # Save model
    import joblib
    joblib.dump(model, 'D:/PythonProject/Result/Model/xgboost_uam_advanced_model.pkl')
    # Save probabilities
    test_results = test_data.copy()
    for i, c in enumerate(le.classes_):
        test_results[f'prob_{c}'] = y_test_proba[:, i]
    test_results.to_csv('D:/PythonProject/Result/Model/synthetic_data_uam_advanced_probabilities.csv', index=False)
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('D:/PythonProject/Result/Feature_Importance/feature_importance_XGBoost_UAM_Prediction.csv', index=False)
    # Save summary
    with open('D:/PythonProject/Result/Model/xgboost_uam_advanced_summary.txt', 'w') as f:
        f.write(f"Best parameters: {model.get_params()}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Training accuracy: {train_acc:.4f}\n")
        f.write(f"Synthetic test accuracy: {test_acc:.4f}\n")
        f.write(f"UAM mean probability: {uam_probs.mean():.4f}\n")
    print("All results saved!")

def main():
    train_data, test_data, le = load_and_prepare_data()
    X_train, X_val, y_train, y_val = split_train_val(train_data)
    best_params, best_cv_score = hyperparameter_tuning(X_train, y_train)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    evaluate_and_save(model, X_train, y_train, X_val, y_val, le, test_data)

if __name__ == "__main__":
    main() 