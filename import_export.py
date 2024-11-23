# Import necessary libraries
import pandas as pd

# Define a function to load and examine the data
def load_and_examine_data(filepath):
    # Load the dataset using Pandas
    data = pd.read_csv(filepath)
    
    # Display basic information about the data
    print("Data Loaded Successfully\n")
    
    # Display the first few rows of the dataset
    print("First 5 Rows of Data:")
    print(data.head(), "\n")
    
    # Show the number of rows and columns
    print(f"Number of Rows: {data.shape[0]}, Number of Columns: {data.shape[1]}\n")
    
    # Show column names and data types
    print("Column Names and Data Types:")
    print(data.dtypes, "\n")
    
    # Check for missing values
    print("Missing Values in Each Column:")
    print(data.isnull().sum(), "\n")
    
    # Display basic statistical information about numeric columns
    print("Statistical Summary of Numeric Columns:")
    print(data.describe())

# Specify the path to your dataset file
filepath = 'your_dataset.csv'  # Replace with your actual file path

# Call the function
load_and_examine_data('Churn_Modelling.csv')
