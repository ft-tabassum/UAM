"""
Count Unique Respondents in AFT Stated Preference Survey
This script counts how many unique people participated and shows
the distribution of responses per person.
"""

import pandas as pd
import numpy as np

def count_unique_respondents():
    """Count unique respondents and analyze response distribution"""
    
    print("="*60)
    print("COUNTING UNIQUE RESPONDENTS IN AFT SURVEY")
    print("="*60)
    
    # Load data
    file_path = 'D:/Files_D/Study/==Thesis==/new_data/aft_2ndversion.xlsx'
    
    try:
        data = pd.read_excel(file_path)
        print(f"[OK] Data loaded successfully")
        print(f"Total rows: {len(data)}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return
    
    # Check if sys_RespNum column exists
    if 'sys_RespNum' not in data.columns:
        print("[ERROR] sys_RespNum column not found!")
        print("Available columns:")
        for i, col in enumerate(data.columns, 1):
            print(f"  {i:2d}. {col}")
        return
    
    # Count unique respondents
    unique_respondents = data['sys_RespNum'].nunique()
    total_responses = len(data)
    
    print(f"\n1. UNIQUE RESPONDENTS COUNT:")
    print(f"   - Unique people (sys_RespNum): {unique_respondents}")
    print(f"   - Total responses: {total_responses}")
    print(f"   - Average responses per person: {total_responses/unique_respondents:.1f}")
    
    # Analyze response distribution per person
    print(f"\n2. RESPONSES PER PERSON:")
    responses_per_person = data['sys_RespNum'].value_counts().sort_index()
    
    print(f"   - Min responses per person: {responses_per_person.min()}")
    print(f"   - Max responses per person: {responses_per_person.max()}")
    print(f"   - Mean responses per person: {responses_per_person.mean():.1f}")
    
    # Show distribution of responses per person
    response_counts = responses_per_person.value_counts().sort_index()
    print(f"\n   Distribution of responses per person:")
    for num_responses, count in response_counts.items():
        print(f"     {num_responses} responses: {count} people")
    
    # Show first few examples
    print(f"\n3. FIRST 10 RESPONDENTS (showing their response counts):")
    for i, (respondent_id, count) in enumerate(responses_per_person.head(10).items()):
        print(f"   Respondent {respondent_id}: {count} responses")
    
    # Check if we have exactly 248 unique people
    print(f"\n4. VERIFICATION:")
    if unique_respondents == 248:
        print(f"   [CONFIRMED] You have exactly 248 unique people as expected!")
    else:
        print(f"   [NOTE] You have {unique_respondents} unique people, not 248")
        print(f"   Expected: 248 people")
        print(f"   Actual: {unique_respondents} people")
        print(f"   Difference: {unique_respondents - 248}")
    
    # Show choice distribution by respondent
    print(f"\n5. CHOICE DISTRIBUTION BY RESPONDENT:")
    choice_by_respondent = data.groupby('sys_RespNum')['CHOICE'].value_counts().unstack(fill_value=0)
    choice_by_respondent.columns = ['Car', 'Public Transport', 'Flying Taxi']
    
    print(f"   Average choices per person:")
    for choice in choice_by_respondent.columns:
        avg_choices = choice_by_respondent[choice].mean()
        print(f"     {choice}: {avg_choices:.1f} times per person")
    
    # Show some examples of individual choice patterns
    print(f"\n6. EXAMPLE INDIVIDUAL CHOICE PATTERNS:")
    for i, (respondent_id, row) in enumerate(choice_by_respondent.head(5).iterrows()):
        print(f"   Respondent {respondent_id}:")
        for choice, count in row.items():
            if count > 0:
                print(f"     {choice}: {count} times")
    
    return unique_respondents, total_responses

if __name__ == "__main__":
    count_unique_respondents()
