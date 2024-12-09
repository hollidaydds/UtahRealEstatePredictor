import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(template_df, num_synthetic_records=150000):
    """Generate synthetic records based on patterns in template data"""
    
    # Create empty dataframe with same columns
    synthetic_df = pd.DataFrame(columns=template_df.columns)
    
    # Get unique values and distributions
    unique_cities = template_df['city'].unique()
    unique_property_types = template_df['propertyType'].unique()
    unique_statuses = template_df['status'].unique()
    
    # Calculate distributions for numeric columns
    price_mean = template_df['price'].mean()
    price_std = template_df['price'].std()
    bedrooms_dist = template_df['bedrooms'].value_counts(normalize=True)
    bathrooms_dist = template_df['bathrooms'].value_counts(normalize=True)
    
    # Get lat/long bounds
    lat_min, lat_max = template_df['latitude'].min(), template_df['latitude'].max()
    long_min, long_max = template_df['longitude'].min(), template_df['longitude'].max()
    
    # Generate synthetic records
    synthetic_records = []
    base_date = datetime.now()
    
    print("Generating synthetic records...")
    for i in range(num_synthetic_records):
        if i % 10000 == 0:
            print(f"Generated {i:,} records...")
            
        # Generate a random date within the last year
        days_ago = random.randint(0, 365)
        current_date = base_date - timedelta(days=days_ago)
        
        # Generate price with some randomness but following distribution
        price = abs(np.random.normal(price_mean, price_std))
        
        # Randomly select bedrooms and bathrooms based on original distribution
        bedrooms = np.random.choice(bedrooms_dist.index, p=bedrooms_dist.values)
        bathrooms = np.random.choice(bathrooms_dist.index, p=bathrooms_dist.values)
        
        # Generate square footage based on bedrooms (rough estimate)
        sq_ft = int(bedrooms * random.uniform(500, 800))
        
        # Generate lot size (typical ranges)
        lot_size = random.uniform(2000, 20000)
        
        # Random year built (mostly recent with some older)
        if random.random() < 0.7:  # 70% newer homes
            year_built = random.randint(1990, 2023)
        else:  # 30% older homes
            year_built = random.randint(1920, 1989)
        
        # Random location within bounds
        latitude = random.uniform(lat_min, lat_max)
        longitude = random.uniform(long_min, long_max)
        
        # Create synthetic record
        record = {
            'id': f'SYNTH_{i}',
            'formattedAddress': f'{random.randint(100, 9999)} {random.choice(["Main", "State", "Center", "Park"])} St',
            'addressLine1': f'{random.randint(100, 9999)} {random.choice(["Main", "State", "Center", "Park"])} St',
            'addressLine2': '',
            'city': np.random.choice(unique_cities),
            'state': 'UT',
            'zipCode': random.randint(84000, 84999),
            'county': random.choice(['Salt Lake', 'Utah', 'Davis', 'Weber', 'Washington']),
            'latitude': latitude,
            'longitude': longitude,
            'propertyType': np.random.choice(unique_property_types),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'squareFootage': sq_ft,
            'lotSize': lot_size,
            'yearBuilt': year_built,
            'status': np.random.choice(unique_statuses),
            'price': price,
            'listedDate': current_date.strftime('%Y-%m-%d'),
            'removedDate': '',
            'createdDate': current_date.strftime('%Y-%m-%d'),
            'lastSeenDate': (current_date + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            'daysOnMarket': random.randint(1, 180)
        }
        
        synthetic_records.append(record)
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_records)
    
    # Combine with original data
    combined_df = pd.concat([template_df, synthetic_df], ignore_index=True)
    
    return combined_df

def main():
    # Read the cleaned data as template
    print("Reading template data...")
    template_df = pd.read_csv('utah_sales_no_outliers_from_zip.csv')
    print(f"Template shape: {template_df.shape}")
    
    # Generate synthetic data
    num_synthetic = 150000  # Generate 150k synthetic records
    combined_df = generate_synthetic_data(template_df, num_synthetic)
    
    # Save combined dataset
    output_file = 'utah_sales_with_synthetic.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nFinal dataset shape: {combined_df.shape}")
    print(f"Saved combined data to: {output_file}")
    
    # Print summary statistics
    print("\nSummary statistics for key columns:")
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'lotSize', 'yearBuilt', 'daysOnMarket']
    print(combined_df[numeric_cols].describe())
    
    # Print sample of synthetic records
    print("\nSample of synthetic records (first 5):")
    print(combined_df[combined_df['id'].str.startswith('SYNTH_')].head())

if __name__ == "__main__":
    main()
