import pandas as pd
import numpy as np
import os

def calculate_monthly_income(income):
    """
    Calculate Monthly_Income using the formula:
    Monthly_Income = income * (1+0.0346)^13 * (1+0.2532)
    Where:
    - Inflation Rate = 25.32%
    - Annual Growth Rate = 3.46%
    - Years = 13 (from 2011 to 2024)
    """
    if pd.isna(income):
        return np.nan
    
    # Formula: income * (1 + annual_growth_rate)^years * (1 + inflation_rate)
    annual_growth_rate = 0.0346
    inflation_rate = 0.2532
    years = 13
    
    monthly_income = income * ((1 + annual_growth_rate) ** years) * (1 + inflation_rate)
    return monthly_income

def categorize_monthly_income(monthly_income):
    """
    Categorize monthly income into predefined categories
    """
    if pd.isna(monthly_income) or monthly_income <= 0:
        return 0  # 'I prefer not to answer'
    
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
    """
    Apply mapping to categorical variables according to the documentation
    """
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
    gender_map = {1: 2, 2: 1, 'male': 2, 'female': 1, 'Male': 2, 'Female': 1, 'Diverse': 3}
    df['gender'] = df['gender'].map(gender_map).fillna(3).astype(int)  # Default to Diverse (3)

    # --- Occupation mapping ---
    print("  - Mapping occupation...")
    # 1=employed, 2=unemployed, 3=student
    occupation_map = {
        'employed': 1, 'Employed': 1, 1: 1,
        'unemployed': 2, 'Unemployed': 2, 2: 2,
        'student': 3, 'Student': 3, 3: 3
    }
    df['occupation'] = df['occupation'].map(occupation_map).fillna(0).astype(int)

    # --- driversLicense mapping ---
    print("  - Mapping driversLicense...")
    df['driversLicense'] = df['driversLicense'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # --- Disability mapping ---
    print("  - Mapping disability...")
    df['disability'] = df['disability'].map({0: 0, 1: 1, '0': 0, '1': 1}).fillna(0).astype(int)

    # --- NEW: Purpose mapping with new categories ---
    print("  - Mapping purpose with new categories...")
    # First, convert old purpose codes to new categories
    # 1=HBW, 2=HBE, 3=HBS, 4=HBR, 5=HBO, 6=NHBW, 7=NHBO
    purpose_conversion = {
        1: 'Business trip',    # HBW -> Business trip
        2: 'Business trip',    # HBE -> Business trip  
        3: 'Visiting family or friends',  # HBS -> Visiting family or friends
        4: 'Tourism',          # HBR -> Tourism
        5: 'Other',            # HBO -> Other
        6: 'Business trip',    # NHBW -> Business trip
        7: 'Other'             # NHBO -> Other
    }
    
    # Convert old purpose codes to new categories
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
    
    return df

def main():
    print("=== INCOME CALCULATION AND MAPPING SCRIPT (UPDATED) ===")
    print("=" * 60)
    
    # Input and output file paths
    input_file = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/microdata_trips.csv"
    output_file = "D:/Thesis/UAM/Result/Vertiport_analysis/Model_XgBoost/Synthetic_population/microdata_trips_with_income&mapping.csv"
    
    try:
        # Read the input data
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Input data shape: {df.shape}")
        print(f"Input columns: {list(df.columns)}")
        print()
        
        # Check if 'income' column exists
        if 'income' not in df.columns:
            print("Error: 'income' column not found in the input data!")
            print(f"Available columns: {list(df.columns)}")
            return
        
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
        df['Monthly_Income'] = df['Monthly_Income_value'].apply(categorize_monthly_income)
        
        # Remove the original 'income' and 'Annual_Income' columns
        df = df.drop(columns=['income', 'Annual_Income'])
        
        # Reorder columns to put Monthly_Income and Monthly_Income_Category after driversLicense
        if 'driversLicense' in df.columns:
            col_list = list(df.columns)
            drivers_license_idx = col_list.index('driversLicense')
            # Remove Monthly_Income_value and Monthly_Income from current position
            col_list.remove('Monthly_Income_value')
            col_list.remove('Monthly_Income')
            # Insert after driversLicense
            col_list.insert(drivers_license_idx + 1, 'Monthly_Income_value')
            col_list.insert(drivers_license_idx + 2, 'Monthly_Income')
            df = df[col_list]
        
        print(f"Income calculation and categorization completed.")
        print(f"Sample Monthly_Income values:")
        print(df[['Monthly_Income_value', 'Monthly_Income']].head(10))
        print()
        
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
        
        # Display mapping summary
        print("=== MAPPING SUMMARY ===")
        print("age: 0=missing, 1=1-17, 2=18-29, 3=30-39, 4=40-49, 5=50-59, 6=60-69, 7=70-79, 8=I prefer not to answer")
        print("gender: 1=Female, 2=Male, 3=Diverse")
        print("occupation: 1=employed, 2=unemployed, 3=student")
        print("driversLicense: 1=True, 0=False")
        print("disability: 0=no, 1=yes")
        print("purpose: 0=Business trip, 1=Medical travel, 2=Other, 3=Tourism, 4=Visiting family or friends")
        print("Monthly_Income: 0=I prefer not to answer, 1=No income, 2=Under €1000, 3=€1000-€2000, 4=€2000-€3000, 5=€3000-€4000, 6=€4000-€5000, 7=€5000-€6000, 8=€6000-€7000, 9=€7000+")
        print()
        
        # Display some statistics
        print("=== DATA SUMMARY ===")
        print(f"Total records: {len(df)}")
        print(f"Average Monthly_Income: €{df['Monthly_Income_value'].mean():.2f}")
        
        print(f"Age distribution:")
        age_counts = df['age'].value_counts().sort_index()
        for age_code, count in age_counts.items():
            age_labels = {0: 'missing', 1: '1-17', 2: '18-29', 3: '30-39', 4: '40-49', 5: '50-59', 6: '60-69', 7: '70-79', 8: 'I prefer not to answer'}
            print(f"  Age {age_code} ({age_labels.get(age_code, 'unknown')}): {count:,} records")
        
        print(f"Gender distribution:")
        gender_counts = df['gender'].value_counts().sort_index()
        for gender_code, count in gender_counts.items():
            gender_labels = {1: 'Female', 2: 'Male', 3: 'Diverse'}
            print(f"  Gender {gender_code} ({gender_labels.get(gender_code, 'unknown')}): {count:,} records")
        
        print(f"Purpose distribution:")
        purpose_counts = df['purpose'].value_counts().sort_index()
        for purpose_code, count in purpose_counts.items():
            purpose_labels = {0: 'Business trip', 1: 'Medical travel', 2: 'Other', 3: 'Tourism', 4: 'Visiting family or friends'}
            print(f"  Purpose {purpose_code} ({purpose_labels.get(purpose_code, 'unknown')}): {count:,} records")
        
        print(f"Monthly Income Category distribution:")
        income_counts = df['Monthly_Income'].value_counts().sort_index()
        for income_code, count in income_counts.items():
            income_labels = {
                0: 'I prefer not to answer', 1: 'No income', 2: 'Under €1000', 
                3: '€1000-€2000', 4: '€2000-€3000', 5: '€3000-€4000', 
                6: '€4000-€5000', 7: '€5000-€6000', 8: '€6000-€7000', 9: '€7000+'
            }
            print(f"  Income {income_code} ({income_labels.get(income_code, 'unknown')}): {count:,} records")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the microdata_trips.py script has been run successfully.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 