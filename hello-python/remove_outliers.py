import pandas as pd
import numpy as np
import zipfile

def clean_data(df):
    df_clean = df.copy()
    
    # Convert price columns to numeric
    price_columns = ['price']
    for col in price_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['bedrooms', 'bathrooms', 'lotSize', 'yearBuilt', 'daysOnMarket']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def analyze_column(df, column):
    """Print detailed analysis of a column's data"""
    print(f"\nAnalysis of {column}:")
    print(f"Total values: {len(df[column])}")
    print(f"Non-null values: {df[column].count()}")
    print(f"Null values: {df[column].isnull().sum()}")
    if df[column].dtype in ['int64', 'float64']:
        print(f"Min: {df[column].min():,.2f}")
        print(f"Max: {df[column].max():,.2f}")
        print(f"Mean: {df[column].mean():,.2f}")
        print(f"Median: {df[column].median():,.2f}")
        print("\nValue distribution:")
        print(df[column].value_counts().head())

def remove_outliers(df, columns, iqr_multiplier=3.0):
    df_clean = df.copy()
    total_removed = 0
    
    print("\nDetailed outlier analysis:")
    
    for column in columns:
        # Analyze column before cleaning
        print(f"\n{'-'*50}")
        print(f"Processing {column}")
        analyze_column(df_clean, column)
        
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Count outliers
        outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)]
        n_outliers = len(outliers)
        
        print(f"\nOutlier removal for {column}:")
        print(f"Lower bound: {lower_bound:,.2f}")
        print(f"Upper bound: {upper_bound:,.2f}")
        print(f"Values outside bounds: {n_outliers:,} ({(n_outliers/len(df_clean)*100):.1f}%)")
        
        if n_outliers > 0:
            print("\nExample outliers:")
            print(outliers[column].head())
        
        # Remove outliers
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        total_removed += n_outliers
    
    print(f"\n{'-'*50}")
    print(f"Total records removed as outliers: {total_removed:,}")
    return df_clean

def main():
    zip_file = "utah_sales_20241203_201859.zip"
    
    # Read CSV from zip file
    print(f"Reading CSV from zip file: {zip_file}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get the name of the CSV file inside the zip
        csv_name = zip_ref.namelist()[0]
        print(f"Found CSV in zip: {csv_name}")
        
        # Read the CSV directly from the zip file
        with zip_ref.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file, low_memory=False)
    
    print(f"Original shape: {df.shape}")
    
    # Clean data first
    print("\nCleaning data types...")
    df_cleaned = clean_data(df)
    
    # Print column info before cleaning
    print("\nColumns in dataset:")
    for col in df_cleaned.columns:
        print(f"{col}: {df_cleaned[col].dtype}")
    
    # Columns to check for outliers
    numeric_columns = [
        'price',
        'bedrooms',
        'bathrooms',
        'lotSize',
        'yearBuilt',
        'daysOnMarket'
    ]
    
    # Remove outliers
    print("\nRemoving outliers...")
    df_no_outliers = remove_outliers(df_cleaned, numeric_columns)
    print(f"\nFinal shape after removing outliers: {df_no_outliers.shape}")
    
    # Calculate percentage of data retained
    retention_rate = (df_no_outliers.shape[0] / df.shape[0]) * 100
    print(f"Retained {retention_rate:.1f}% of original data")
    
    # Print summary statistics after outlier removal
    print("\nSummary statistics after outlier removal:")
    print(df_no_outliers[numeric_columns].describe())
    
    # Save to new CSV
    output_file = "utah_sales_no_outliers_from_zip.csv"
    df_no_outliers.to_csv(output_file, index=False)
    print(f"\nSaved cleaned data without outliers to: {output_file}")

if __name__ == "__main__":
    main()
