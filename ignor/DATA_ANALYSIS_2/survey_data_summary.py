import pandas as pd
import numpy as np

print("STATED PREFERENCE SURVEY DATA SUMMARY")
print("=" * 50)

# Load the survey data
data = pd.read_csv('/Result/DataPreprocessing_aft/aft_normalized.csv')

print(f"Survey Dataset: {len(data):,} observations")
print(f"Total variables: {len(data.columns)}")

# Choice distribution (this is the key outcome of your survey)
print(f"\nSURVEY CHOICE DISTRIBUTION")
print("=" * 30)
choice_dist = data['CHOICE'].value_counts().sort_index()
choice_names = {0: 'Car', 1: 'Public Transport', 2: 'Autonomous Flying Taxi'}

for choice, count in choice_dist.items():
    mode_name = choice_names.get(choice, f'Choice {choice}')
    percentage = (count / len(data)) * 100
    print(f"{mode_name}: {count:,} ({percentage:.1f}%)")

# Survey respondent characteristics
print(f"\nSURVEY RESPONDENT CHARACTERISTICS")
print("=" * 40)

# Gender distribution
if 'female' in data.columns:
    female_count = data['female'].sum()
    male_count = len(data) - female_count
    print(f"Gender Distribution:")
    print(f"  Female: {female_count:,} ({female_count/len(data)*100:.1f}%)")
    print(f"  Male: {male_count:,} ({male_count/len(data)*100:.1f}%)")

# Age distribution
age_columns = [col for col in data.columns if col.startswith('age_')]
if age_columns:
    print(f"\nAge Distribution:")
    for age_col in age_columns:
        count = data[age_col].sum()
        percentage = (count / len(data)) * 100
        age_group = age_col.replace('age_', '').replace('-', '-')
        print(f"  {age_group}: {count:,} ({percentage:.1f}%)")

# Employment status
employment_columns = [col for col in data.columns if col.startswith('employment_')]
if employment_columns:
    print(f"\nEmployment Status:")
    for emp_col in employment_columns:
        count = data[emp_col].sum()
        percentage = (count / len(data)) * 100
        emp_type = emp_col.replace('employment_', '').title()
        print(f"  {emp_type}: {count:,} ({percentage:.1f}%)")

# Education level
education_columns = [col for col in data.columns if col.startswith('education_')]
if education_columns:
    print(f"\nEducation Level:")
    for edu_col in education_columns:
        count = data[edu_col].sum()
        percentage = (count / len(data)) * 100
        edu_type = edu_col.replace('education_', '').title()
        print(f"  {edu_type}: {count:,} ({percentage:.1f}%)")

# Car ownership
car_columns = [col for col in data.columns if col.startswith('car_')]
if car_columns:
    print(f"\nCar Ownership:")
    for car_col in car_columns:
        count = data[car_col].sum()
        percentage = (count / len(data)) * 100
        car_count = car_col.replace('car_', '').replace('andmore', '+')
        print(f"  {car_count} car(s): {count:,} ({percentage:.1f}%)")

# Children in household
child_columns = [col for col in data.columns if col.startswith('child_household_')]
if child_columns:
    print(f"\nChildren in Household:")
    for child_col in child_columns:
        count = data[child_col].sum()
        percentage = (count / len(data)) * 100
        child_count = child_col.replace('child_household_', '').replace('andmore', '+')
        print(f"  {child_count} child(ren): {count:,} ({percentage:.1f}%)")

# Current transportation mode
if 'current_transportmode' in data.columns:
    print(f"\nCurrent Transportation Mode:")
    mode_dist = data['current_transportmode'].value_counts().sort_index()
    mode_names = {1: 'Car', 2: 'Public Transport', 3: 'Other'}
    for mode, count in mode_dist.items():
        mode_name = mode_names.get(mode, f'Mode {mode}')
        percentage = (count / len(data)) * 100
        print(f"  {mode_name}: {count:,} ({percentage:.1f}%)")

# Driving license
if 'driving_license_yes' in data.columns:
    license_yes = data['driving_license_yes'].sum()
    license_no = len(data) - license_yes
    print(f"\nDriving License:")
    print(f"  Yes: {license_yes:,} ({license_yes/len(data)*100:.1f}%)")
    print(f"  No: {license_no:,} ({license_no/len(data)*100:.1f}%)")

# Commuting status
if 'Commuting' in data.columns:
    commuting_yes = data['Commuting'].sum()
    commuting_no = len(data) - commuting_yes
    print(f"\nCommuting Status:")
    print(f"  Commuting: {commuting_yes:,} ({commuting_yes/len(data)*100:.1f}%)")
    print(f"  Non-commuting: {commuting_no:,} ({commuting_no/len(data)*100:.1f}%)")

# Adults in household
adult_columns = [col for col in data.columns if col.startswith('adults_household_')]
if adult_columns:
    print(f"\nAdults in Household:")
    for adult_col in adult_columns:
        count = data[adult_col].sum()
        percentage = (count / len(data)) * 100
        adult_count = adult_col.replace('adults_household_', '').replace('andmore', '+')
        print(f"  {adult_count} adult(s): {count:,} ({percentage:.1f}%)")

# Travel time preferences (alternative-specific attributes)
print(f"\nTRAVEL TIME PREFERENCES (Alternative-Specific Attributes)")
print("=" * 50)

time_columns = [col for col in data.columns if 'TT' in col]
if time_columns:
    for time_col in time_columns:
        mode = time_col.split('_')[0]
        mode_name = {'CAR': 'Car', 'PT': 'Public Transport', 'AFT': 'Autonomous Flying Taxi'}.get(mode, mode)
        values = data[time_col].dropna()
        print(f"{mode_name} Travel Time:")
        print(f"  Mean: {values.mean():.1f} minutes")
        print(f"  Range: {values.min():.0f} - {values.max():.0f} minutes")
        print(f"  Std Dev: {values.std():.1f} minutes")

# Travel cost preferences
print(f"\nTRAVEL COST PREFERENCES (Alternative-Specific Attributes)")
print("=" * 50)

cost_columns = [col for col in data.columns if 'CO' in col and 'TT' not in col]
if cost_columns:
    for cost_col in cost_columns:
        mode = cost_col.split('_')[0]
        mode_name = {'CAR': 'Car', 'PT': 'Public Transport', 'AFT': 'Autonomous Flying Taxi'}.get(mode, mode)
        values = data[cost_col].dropna()
        print(f"{mode_name} Travel Cost:")
        print(f"  Mean: €{values.mean():.2f}")
        print(f"  Range: €{values.min():.2f} - €{values.max():.2f}")
        print(f"  Std Dev: €{values.std():.2f}")

# Attitude variables (Likert scale responses)
print(f"\nATTITUDE VARIABLES (Likert Scale Responses)")
print("=" * 40)

# Likelihood variables
likelihood_cols = [col for col in data.columns if col.startswith('Likelihood_')]
if likelihood_cols:
    print(f"Likelihood Variables (0-5 scale):")
    for col in likelihood_cols:
        values = data[col].dropna()
        print(f"  {col}: Mean={values.mean():.2f}, Std={values.std():.2f}")

# Attitude to AFT
attitude_cols = [col for col in data.columns if col.startswith('AtoLattitude_')]
if attitude_cols:
    print(f"\nAttitude to AFT Variables (0-5 scale):")
    for col in attitude_cols:
        values = data[col].dropna()
        print(f"  {col}: Mean={values.mean():.2f}, Std={values.std():.2f}")

# Technology concern
tech_cols = [col for col in data.columns if col.startswith('technologyconcern_')]
if tech_cols:
    print(f"\nTechnology Concern Variables (0-4 scale):")
    for col in tech_cols:
        values = data[col].dropna()
        print(f"  {col}: Mean={values.mean():.2f}, Std={values.std():.2f}")

# Environment concern
env_cols = [col for col in data.columns if col.startswith('environmentconcern_')]
if env_cols:
    print(f"\nEnvironment Concern Variables (0-4 scale):")
    for col in env_cols:
        values = data[col].dropna()
        print(f"  {col}: Mean={values.mean():.2f}, Std={values.std():.2f}")

# Satisfaction
if 'satisfaction' in data.columns:
    values = data['satisfaction'].dropna()
    print(f"\nSatisfaction (0-4 scale):")
    print(f"  Mean: {values.mean():.2f}, Std: {values.std():.2f}")

# Data quality summary
print(f"\nSURVEY DATA QUALITY")
print("=" * 20)
print(f"Total responses: {len(data):,}")
print(f"Complete responses: {data.dropna().shape[0]:,}")
print(f"Missing values: {data.isnull().sum().sum():,}")
print(f"Data completeness: {((len(data) * len(data.columns) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%")

# Summary for report
print(f"\nSUMMARY FOR YOUR REPORT")
print("=" * 30)
print(f"• Survey sample: {len(data):,} respondents")
print(f"• Choice distribution: Car (39.1%), PT (44.9%), AFT (16.0%)")
print(f"• Data completeness: {((len(data) * len(data.columns) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%")
print(f"• Survey includes: demographic, attitudinal, and alternative-specific attributes")
print(f"• Variables: {len(data.columns)} total variables")
print(f"• Target: stated choice among 3 transportation modes")

print("\n" + "="*50)
print("SURVEY DATA SUMMARY COMPLETE!")
print("="*50)
