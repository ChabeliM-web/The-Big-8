"""
This script loads and preprocesses user and school review data from CSV files.

Steps:
1. The 'load_user_data' function loads user reviews from a specified CSV file into a Pandas DataFrame.
2. The 'load_school_data' function loads school data from a specified CSV file into a Pandas DataFrame.
3. Missing values in both user and school data are handled by filling them with empty strings.
4. A sample of the user and school data is printed to provide an overview of the data structure.
5. If the file is not found, an error message is displayed, and the function returns None.

Dependencies:
- pandas for data manipulation

Parameters:
- 'generate_user_reviews_with_ids.csv' contains user review data.
- 'gauteng_schools.csv' contains school information.

Returns:
- Preprocessed user and school data.
"""


import pandas as pd

def load_user_data(file_path):
    """Load user review data from a CSV file."""
    try:
        user_data = pd.read_csv(file_path)
        print("User data loaded successfully.")
        return user_data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def load_school_data(file_path):
    """Load school data from a CSV file."""
    try:
        school_data = pd.read_csv(file_path)
        print("School data loaded successfully.")
        return school_data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def preprocess_data(user_data, school_data):
    """Preprocess user and school data for training."""
    user_data.fillna('', inplace=True)
    school_data.fillna('', inplace=True)
    
    print("\nUser Data Sample:")
    print(user_data.head())
    print("\nSchool Data Sample:")
    print(school_data.head())
    
    return user_data, school_data

if __name__ == "__main__":
    # Load the data
    user_data = load_user_data('data/generate_user_reviews_with_ids.csv')
    school_data = load_school_data('data/gauteng_schools.csv')

    if user_data is not None and school_data is not None:
        # Preprocess the data
        processed_user_data, processed_school_data = preprocess_data(user_data, school_data)
