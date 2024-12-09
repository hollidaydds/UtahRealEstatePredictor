import pandas as pd
import os
import glob
from datetime import datetime

def get_latest_csv():
    # Get list of all CSV files matching our pattern
    csv_files = glob.glob('utah_sales_*.csv')
    if not csv_files:
        raise FileNotFoundError("No utah_sales CSV files found in the current directory")
    
    # Get the most recent file based on file modification time
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"Processing file: {latest_file}")
    return latest_file

def clean_dataframe(df):
    try:
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Convert price columns to numeric, removing any currency symbols and commas
        price_columns = ['price', 'listPrice']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Convert date columns to datetime
        date_columns = ['createdDate', 'lastSeen', 'listDate']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean up boolean columns
        bool_columns = ['isForeclosure', 'isNewConstruction', 'isPriceReduced', 'isTaxDistressed']
        for col in bool_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'true': True, 'false': False})
        
        # Convert numeric columns
        numeric_columns = ['bathrooms', 'bedrooms', 'livingArea', 'lotSize', 'yearBuilt']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle text columns
        text_columns = ['description', 'propertyType', 'city', 'state', 'zipCode']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('')
                df_clean[col] = df_clean[col].astype(str)
        
        # Remove any completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Fill NaN values appropriately
        for col in bool_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(False)
        
        # Add derived columns
        if 'price' in df_clean.columns and 'livingArea' in df_clean.columns:
            df_clean['pricePerSqFt'] = df_clean['price'] / df_clean['livingArea']
        
        if 'listPrice' in df_clean.columns and 'price' in df_clean.columns:
            df_clean['priceDropAmount'] = df_clean['listPrice'] - df_clean['price']
            df_clean['priceDropPercent'] = (df_clean['priceDropAmount'] / df_clean['listPrice']) * 100
        
        return df_clean
    
    except Exception as e:
        print(f"Error in clean_dataframe: {str(e)}")
        raise

def main():
    try:
        # Get the latest CSV file
        latest_csv = get_latest_csv()
        
        # Read the CSV file
        print("Reading CSV file...")
        df = pd.read_csv(latest_csv, low_memory=False)  # Added low_memory=False to handle mixed types
        print(f"Original shape: {df.shape}")
        
        # Print column names
        print("\nColumns in the dataset:")
        print(df.columns.tolist())
        
        # Clean the data
        print("\nCleaning data...")
        df_clean = clean_dataframe(df)
        print(f"Cleaned shape: {df_clean.shape}")
        
        # Save cleaned data
        output_file = f"cleaned_{latest_csv}"
        df_clean.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        # Print summary statistics for numeric columns
        print("\nSummary Statistics (numeric columns):")
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        print(df_clean[numeric_cols].describe())
        
        # Print data types for all columns
        print("\nColumn Data Types:")
        print(df_clean.dtypes)
        
        return df_clean
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    df_clean = main()
