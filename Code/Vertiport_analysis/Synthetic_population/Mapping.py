import pandas as pd
import numpy as np


def calculate_monthly_income(income):
    """ Calculate Monthly_Income using the formula:
    Monthly_Income = income * (1+0.0346)^13 * (1+0.2532)
    Where:
    - Inflation Rate = 25.32%
    - Annual Growth Rate = 3.46%
    - Years = 13 (from 2011 to 2024) """
    if pd.isna(income):
        return np.nan
    
    # Formula: income * (1 + annual_growth_rate)^years * (1 + inflation_rate)
    annual_growth_rate = 0.0346
    inflation_rate = 0.2532
    years = 13
    
    monthly_income = income * ((1 + annual_growth_rate) ** years) * (1 + inflation_rate)
    return monthly_income

def categorize_monthly_income(monthly_income):
    """Categorize monthly income into predefined categories"""
    if pd.isna(monthly_income) or monthly_income <= 0:
        return 0  # 'I prefer not to answer'
    if monthly_income == 0:
        return 1  # 'No income'
    if monthly_income < 1000:
        return 2  # 'Under € 1000'
    elif monthly_income < 2000:
        return 3  # '€ 1000 to less than € 2000'
    elif monthly_income < 3000:
        return 4  # '€ 2000 to less than € 3000'
    elif monthly_income < 4000:
        return 5  # '€ 3000 to less than € 4000'
    elif monthly_income < 5000:
        return 6  # '€ 4000 to less than € 5000'
    elif monthly_income < 6000:
        return 7  # '€ 5000 to less than € 6000'
    elif monthly_income < 7000:
        return 8  # '€ 6000 to less than € 7000'
    else:
        return 9  # '€ 7000 or more'

def apply_mapping(df):
    """ Apply mapping to categorical variables according to the documentation"""
    print("Applying mappings...")
    
    # --- Age binning ---
    print("  - Mapping age...")
    def bin_age(age):
        if pd.isna(age):
            return 0  # 'missing'
        try:
            age = int(age)
        except:
            return 0  # 'missing'
        if 1 <= age <= 17:
            return 1  # '1-17'
        elif 18 <= age <= 29:
            return 2  # '18-29'
        elif 30 <= age <= 39:
            return 3  # '30-39'
        elif 40 <= age <= 49:
            return 4  # '40-49'
        elif 50 <= age <= 59:
            return 5  # '50-59'
        elif 60 <= age <= 69:
            return 6  # '60-69'
        elif 70 <= age <= 79:
            return 7  # '70-79'
        else:
            return 8  # 'I prefer not to answer'

    df['age'] = df['age'].apply(bin_age)

    # --- Gender mapping ---
    print("  - Mapping gender...")
    # 1=Female, 2=Male, 3=Diverse
    gender_map = {'Male': 2, 'Female': 1, 'Diverse': 3}
    df['gender'] = df['gender'].map(gender_map).fillna(3).astype(int)

    # --- Occupation mapping ---
    print("  - Mapping occupation...")
    # First, let's see what occupation values we have
    print(f"  Original occupation unique values: {df['occupation'].unique()}")
    print(f"  Original occupation value counts: {df['occupation'].value_counts().sort_index()}")
    
    # Map numeric occupation codes to our target categories
    # Assuming original codes: 1=employed, 2=unemployed, 3=student, 0/other=prefer not to answer
    occupation_map = {
        0: 0,  # 'I prefer not to answer'
        1: 1,  # 'Employed'
        2: 2,  # 'Unemployed' 
        3: 3,  # 'Student'
        'I prefer not to answer': 0,
        'Employed': 1,
        'Unemployed': 2,
        'Student': 3
    }
    df['occupation'] = df['occupation'].map(occupation_map).fillna(0).astype(int)
    
    print(f"  After mapping occupation unique values: {df['occupation'].unique()}")
    print(f"  After mapping occupation value counts: {df['occupation'].value_counts().sort_index()}")
    
    # Ensure all occupation categories are present (add dummy rows if needed)
    occupation_categories = [0, 1, 2, 3]
    missing_categories = set(occupation_categories) - set(df['occupation'].unique())
    if missing_categories:
        print(f"  Adding dummy rows for missing occupation categories: {missing_categories}")
        # Create dummy rows for missing categories
        dummy_rows = []
        for cat in missing_categories:
            dummy_row = df.iloc[0].copy()  # Copy first row as template
            dummy_row['occupation'] = cat
            dummy_row['trip_id'] = f"dummy_occ_{cat}"  # Unique trip_id for dummy
            dummy_rows.append(dummy_row)
        
        # Add dummy rows to dataframe
        df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
        print(f"  Added {len(dummy_rows)} dummy rows for missing occupation categories")

    # --- driversLicense mapping ---
    print("  - Mapping driversLicense...")
    df['driversLicense'] = df['driversLicense'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # --- Disability mapping ---
    print("  - Mapping disability...")
    # The disability column should already be 0/1, so we just ensure it's integer
    df['disability'] = df['disability'].astype(int)
    
    print(f"  Disability unique values: {df['disability'].unique()}")
    print(f"  Disability value counts: {df['disability'].value_counts().sort_index()}")
    
    # Ensure all disability categories are present (add dummy rows if needed)
    disability_categories = [0, 1]
    missing_disability = set(disability_categories) - set(df['disability'].unique())
    if missing_disability:
        print(f"  Adding dummy rows for missing disability categories: {missing_disability}")
        # Create dummy rows for missing categories
        dummy_rows = []
        for cat in missing_disability:
            dummy_row = df.iloc[0].copy()  # Copy first row as template
            dummy_row['disability'] = cat
            dummy_row['trip_id'] = f"dummy_dis_{cat}"  # Unique trip_id for dummy
            dummy_rows.append(dummy_row)
        
        # Add dummy rows to dataframe
        df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
        print(f"  Added {len(dummy_rows)} dummy rows for missing disability categories")

    # Purpose mapping with new categories ---
    print("  - Mapping purpose with new categories...")
    print(f"  Original purpose unique values: {df['purpose'].unique()}")
    print(f"  Original purpose value counts: {df['purpose'].value_counts().sort_index()}")
    
    # Map string purpose values to new categories
    # HBW, HBE, NHBW = Business trip
    # HBS -> Visiting family or friends  
    # HBR -> Tourism
    # HBO, NHBO -> Other
    purpose_conversion = {
        'HBW': 'Business trip',    # House-based Work
        'HBE': 'Business trip',    # House-based Education
        'NHBW': 'Business trip',   # Non-House-based Work
        'HBS': 'Visiting family or friends',  # House-based Shopping
        'HBR': 'Tourism',          # House-based Recreation
        'HBO': 'Other',            # House-based Other
        'NHBO': 'Other'            # Non-House-based Other
    }
    
    # Convert string purpose values to new categories
    df['purpose_category'] = df['purpose'].map(purpose_conversion).fillna('Other')
    
    # Now map categories to final codes
    purpose_final_map = {
        'Business trip': 0,
        'Medical travel': 1,
        'Other': 2,
        'Tourism': 3,
        'Visiting family or friends': 4
    }
    
    df['purpose'] = df['purpose_category'].map(purpose_final_map).fillna(2).astype(int)
    
    # Remove the temporary category column
    df = df.drop(columns=['purpose_category'])
    
    print(f"  After mapping purpose unique values: {df['purpose'].unique()}")
    print(f"  After mapping purpose value counts: {df['purpose'].value_counts().sort_index()}")
    
    # Ensure all purpose categories are present (add dummy rows if needed)
    purpose_categories = [0, 1, 2, 3, 4]  # Business trip, Medical travel, Other, Tourism, Visiting family or friends
    missing_purpose = set(purpose_categories) - set(df['purpose'].unique())
    if missing_purpose:
        print(f"  Adding dummy rows for missing purpose categories: {missing_purpose}")
        # Create dummy rows for missing categories
        dummy_rows = []
        for cat in missing_purpose:
            dummy_row = df.iloc[0].copy()  # Copy first row as template
            dummy_row['purpose'] = cat
            dummy_row['trip_id'] = f"dummy_purpose_{cat}"  # Unique trip_id for dummy
            dummy_rows.append(dummy_row)
        
        # Add dummy rows to dataframe
        df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
        print(f"  Added {len(dummy_rows)} dummy rows for missing purpose categories")
    
    return df

def main():
    print("=== INCOME CALCULATION AND MAPPING SCRIPT  ===")
    print("=" * 60)
    
    # Input and output file paths
    input_file = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/microdata_trips.csv"
    output_file = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/Mapping.csv"
    
    try:
        # Read the input data
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Input data shape: {df.shape}")
        print(f"Input columns: {list(df.columns)}")
        print()
        
        # Calculate Monthly_Income (this is actually annual income, we need to convert to monthly)
        print("Calculating Annual Income (adjusted for inflation and growth)...")
        print("Formula: Annual_Income = income * (1+0.0346)^13 * (1+0.2532)")
        print("Where: Inflation Rate = 25.32%, Annual Growth Rate = 3.46%, Years = 13")
        
        df['Annual_Income'] = df['income'].apply(calculate_monthly_income)
        
        # Convert annual income to monthly income
        print("Converting Annual Income to Monthly Income...")
        df['Monthly_Income_value'] = df['Annual_Income'] / 12.0
        
        # Categorize monthly income
        print("Categorizing Monthly Income...")
        df['monthly_income'] = df['Monthly_Income_value'].apply(categorize_monthly_income)
        
        # Remove the original 'income' and 'Annual_Income' columns
        df = df.drop(columns=['income', 'Annual_Income'])
        
        # Reorder columns to put Monthly_Income and Monthly_Income_Category after driversLicense
        if 'driversLicense' in df.columns:
            col_list = list(df.columns)
            drivers_license_idx = col_list.index('driversLicense')

            # Remove Monthly_Income_value and Monthly_Income from current position
            col_list.remove('Monthly_Income_value')
            col_list.remove('monthly_income')
            # Insert after driversLicense
            col_list.insert(drivers_license_idx + 1, 'monthly_income')
            df = df[col_list]
        
        # Apply mappings
        df = apply_mapping(df)
        
        # Save the processed data
        print(f"Saving output file: {output_file}")
        df.to_csv(output_file, index=False)
        
        print()
        print("=== PROCESSING COMPLETED ===")
        print(f"Output file: {output_file}")
        print(f"Output shape: {df.shape}")
        print(f"Output columns: {list(df.columns)}")
        print()

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the microdata_trips.py script has been run successfully.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 