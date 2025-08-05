# SIMPLE FIX: Replace your preprocess_for_ml function with this version

def preprocess_for_ml(df):
    print("=== DATA PROCESSING SCRIPT ===")
    print("=" * 60)
    print("Available columns in DataFrame:")
    print(df.columns.tolist())
    print()

    # Define columns to keep for processing
    features_to_keep = [
        'person_id', 'age', 'gender', 'child_household', 'occupation', 'adults_household', 'driversLicense',
        'Monthly_Income', 'disability', 'purpose', 'autos', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
        'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto',
        'travel_cost_pt'
    ]

    # Create a copy with only the features we want to process
    data_processed = df[features_to_keep].copy()

    # Define ordinal and nominal columns
    ordinal_cols = ['Monthly_Income', 'age', 'gender', 'purpose']
    nominal_cols = ['occupation', 'driversLicense', 'autos', 'child_household', 'adults_household']
    numerical_cols = ['disability', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
        'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto',
                      'travel_cost_pt']

    # Handle missing values
    data_processed[nominal_cols] = data_processed[nominal_cols].fillna('0').astype(str)
    for col in ordinal_cols:
        data_processed[col] = data_processed[col].fillna(0)
    for col in numerical_cols:
        data_processed[col] = data_processed[col].fillna(0)

    # **KEY FIX**: Check what occupation categories exist
    print(f"Current occupation categories: {sorted(data_processed['occupation'].unique())}")
    
    # **CRITICAL FIX**: Use categories parameter in OneHotEncoder to ensure all categories are created
    # Define all possible categories
    all_occupation_categories = ['0', '1', '2', '3']  # All possible occupation values
    all_driversLicense_categories = ['0', '1']
    
    # Label Encoding for ordinal columns
    label_encoders = {}
    for col in ordinal_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
        label_encoders[col] = le

    # One-Hot Encoding for nominal columns WITH EXPLICIT CATEGORIES
    if nominal_cols:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Create OneHotEncoder with explicit categories for occupation and driversLicense
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(
                categories=[
                    all_occupation_categories,    # occupation categories
                    all_driversLicense_categories, # driversLicense categories  
                    'auto',  # autos - auto-detect from data
                    'auto',  # child_household - auto-detect from data
                    'auto'   # adults_household - auto-detect from data
                ],
                handle_unknown='ignore', 
                sparse_output=False  # Ensure dense output
            ))
        ])
        
        preprocessor = ColumnTransformer(transformers=[('nominal', categorical_transformer, nominal_cols)],
                                         remainder='passthrough')
        X_processed = preprocessor.fit_transform(data_processed)

        # Get one-hot encoded feature names
        onehot_feature_names = list(
            preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_cols)
        )

        # Get the remaining column names (passthrough columns)
        passthrough_cols = [col for col in data_processed.columns if col not in nominal_cols]

        # Combine all feature names
        feature_names = onehot_feature_names + passthrough_cols

    else:
        X_processed = data_processed.copy()
        feature_names = list(data_processed.columns)

    # Convert to dense array if sparse
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    # Convert to DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Verify all occupation columns exist
    occupation_cols = [col for col in X_processed_df.columns if col.startswith('occupation_')]
    print(f"Generated occupation columns: {occupation_cols}")
    for col in occupation_cols:
        count = X_processed_df[col].sum()
        print(f"  {col}: {count} occurrences")
    
    return X_processed_df


# ALTERNATIVE APPROACH: If the above doesn't work, use this manual approach
def preprocess_for_ml_manual_approach(df):
    print("=== DATA PROCESSING SCRIPT (MANUAL APPROACH) ===")
    print("=" * 60)
    
    # [Previous setup code remains the same...]
    features_to_keep = [
        'person_id', 'age', 'gender', 'child_household', 'occupation', 'adults_household', 'driversLicense',
        'Monthly_Income', 'disability', 'purpose', 'autos', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
        'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto',
        'travel_cost_pt'
    ]

    data_processed = df[features_to_keep].copy()
    
    # Handle missing values and define column types
    ordinal_cols = ['Monthly_Income', 'age', 'gender', 'purpose']
    nominal_cols = ['occupation', 'driversLicense', 'autos', 'child_household', 'adults_household']
    numerical_cols = ['disability', 'distance', 'in_vehicle_time_auto', 'waiting_time_auto',
        'travel_time_auto', 'in_vehicle_time_pt', 'waiting_time_pt', 'travel_time_pt', 'travel_cost_auto',
                      'travel_cost_pt']

    data_processed[nominal_cols] = data_processed[nominal_cols].fillna('0').astype(str)
    for col in ordinal_cols:
        data_processed[col] = data_processed[col].fillna(0)
    for col in numerical_cols:
        data_processed[col] = data_processed[col].fillna(0)

    # Label encode ordinal columns
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in ordinal_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
        label_encoders[col] = le

    # **MANUAL ONE-HOT ENCODING APPROACH**
    final_df = data_processed[['person_id'] + numerical_cols + ordinal_cols].copy()
    
    # Manually create occupation columns (ensure all 4 categories exist)
    for i in range(4):  # 0, 1, 2, 3
        col_name = f'occupation_{i}'
        final_df[col_name] = (data_processed['occupation'] == str(i)).astype(int)
        print(f"Created {col_name}: {final_df[col_name].sum()} occurrences")
    
    # Manually create driversLicense columns
    for i in range(2):  # 0, 1
        col_name = f'driversLicense_{i}'
        final_df[col_name] = (data_processed['driversLicense'] == str(i)).astype(int)
        print(f"Created {col_name}: {final_df[col_name].sum()} occurrences")
    
    # Auto-detect and create columns for other nominal features
    import pandas as pd
    for col in ['autos', 'child_household', 'adults_household']:
        unique_vals = sorted(data_processed[col].unique())
        for val in unique_vals:
            col_name = f'{col}_{val}'
            final_df[col_name] = (data_processed[col] == val).astype(int)
    
    return final_df