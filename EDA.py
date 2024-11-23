# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Data Loaded Successfully")
    return data

# Conduct Exploratory Data Analysis
def exploratory_data_analysis(data):
    print("\n--- Exploratory Data Analysis (EDA) ---\n")
    
    # Statistical Summary
    print("\nStatistical Summary of Numeric Columns:")
    print(data.describe(), "\n")
    
    # Data Info: Data types, non-null values
    print("\nData Information:")
    print(data.info(), "\n")
    
    # Checking for missing values
    print("\nMissing Values in Each Column:")
    print(data.isnull().sum(), "\n")
    
    # Distribution of Churn
    plt.figure(figsize=(6,4))
    sns.countplot(data=data, x='Churn')
    plt.title("Churn Distribution")
    plt.show()
    
    # Distribution of Numerical Features
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns].hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Histograms of Numeric Features")
    plt.show()
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_data = data[numeric_columns]
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()
    
    # Relationship between Categorical Features and Churn
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Churn':  # Avoid repeating the Churn column
            plt.figure(figsize=(6,4))
            sns.countplot(data=data, x=col, hue='Churn')
            plt.title(f"Churn by {col}")
            plt.legend(title="Churn")
            plt.show()

    # Pairplot for Relationships
    sns.pairplot(data[numeric_columns], diag_kind='kde', corner=True)
    plt.suptitle("Pairplot of Numeric Features")
    plt.show()

# Main function
def main():
    filepath = 'Churn_Modelling.csv'  # Replace with your actual file path
    data = load_data(filepath)
    exploratory_data_analysis(data)

if __name__ == "__main__":
    main()
