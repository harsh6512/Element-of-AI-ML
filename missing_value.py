import pandas as pd
import numpy as np

# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Data Loaded Successfully")
    return data

# Handle Missing Values
def handle_missing_values(data):
    # Example: Drop columns with more than 50% missing values
    threshold = 0.5 * data.shape[0]
    data = data.dropna(thresh=threshold, axis=1)
    
    # Fill numeric columns with mean values
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Fill categorical columns with mode values
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
    
    print("Missing values handled.")
    return data

# Detect and Handle Outliers
def handle_outliers(data):
    # Using IQR method for outlier detection
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap the outliers to the lower and upper bounds
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    
    print("Outliers handled using IQR method.")
    return data

# Main function
def main():
    filepath = 'Churn_Modelling.csv'  # Replace with the path to your dataset
    data = load_data(filepath)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Handle outliers
    data = handle_outliers(data)
    
    # Display cleaned data information
    print("\nData after Cleaning:")
    print(data.info())
    print("\nFirst 5 Rows of Cleaned Data:")
    print(data.head())

if __name__ == "__main__":
    main()
