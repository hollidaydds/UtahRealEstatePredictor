import pandas as pd
import pickle
from pathlib import Path
import argparse

# County name to code mapping
COUNTY_CODES = {
    'Salt Lake': 34,
    'Utah': 56,
    'Davis': 48,
    'Weber': 52,
    'Washington': 42,
    'Cache': 10,
    'Box Elder': 4,
    'Iron': 20,
    'Tooele': 44,
    'Summit': 40,
    'Carbon': 8,
    'Beaver': 2,
    'Duchesne': 12,
    'Juab': 22
}

def load_model(model_path):
    """Load the trained model from disk"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_house_price(
    bedrooms: float,
    bathrooms: float,
    square_footage: float,
    lot_size: float,
    year_built: int,
    county: str,
    property_type: str,
    model_path: str = None
) -> float:
    """
    Predict house price using the trained model
    
    Parameters:
    -----------
    bedrooms : float
        Number of bedrooms
    bathrooms : float
        Number of bathrooms
    square_footage : float
        Square footage of the house
    lot_size : float
        Lot size in acres
    year_built : int
        Year the house was built
    county : str
        County name (e.g., 'Salt Lake', 'Utah')
    property_type : str
        Type of property (e.g., 'Single Family')
    model_path : str, optional
        Path to the model file. If None, uses default path
        
    Returns:
    --------
    float
        Predicted price of the house
    """
    # Set default model path if none provided
    if model_path is None:
        model_path = Path(__file__).parent / 'house_price_model.pkl'
    
    # Load the model (which includes the preprocessing pipeline)
    model = load_model(model_path)
    
    # Convert county name to code
    county = county.strip()
    if county not in COUNTY_CODES:
        raise ValueError(f"Unknown county: {county}. Must be one of: {', '.join(COUNTY_CODES.keys())}")
    county_code = COUNTY_CODES[county]
    
    # Create input data with correct column order
    # Note: The order must match what the model expects
    house_data = {
        'bedrooms': [bedrooms],  # Convert to list for DataFrame
        'bathrooms': [bathrooms],
        'squareFootage': [square_footage],
        'lotSize': [lot_size],
        'yearBuilt': [year_built],
        'county': [county_code],  # Use numeric county code
        'propertyType': [property_type]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(house_data)
    
    # The pipeline will handle preprocessing (scaling numeric features and encoding categorical features)
    predicted_price = model.predict(input_df)[0]
    
    return predicted_price

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict house price based on features')
    parser.add_argument('--bedrooms', type=float, required=True, help='Number of bedrooms')
    parser.add_argument('--bathrooms', type=float, required=True, help='Number of bathrooms')
    parser.add_argument('--square-footage', type=float, required=True, help='Square footage of the house')
    parser.add_argument('--lot-size', type=float, required=True, help='Lot size in acres')
    parser.add_argument('--year-built', type=int, required=True, help='Year the house was built')
    parser.add_argument('--county', type=str, required=True, help=f'County name. Must be one of: {", ".join(COUNTY_CODES.keys())}')
    parser.add_argument('--property-type', type=str, required=True, help='Type of property (e.g., "Single Family")')
    parser.add_argument('--model-path', type=str, help='Path to model file (optional)')
    
    args = parser.parse_args()
    
    try:
        predicted_price = predict_house_price(
            bedrooms=args.bedrooms,
            bathrooms=args.bathrooms,
            square_footage=args.square_footage,
            lot_size=args.lot_size,
            year_built=args.year_built,
            county=args.county,
            property_type=args.property_type,
            model_path=args.model_path
        )
        
        print(f"\nPredicted House Price: ${predicted_price:,.2f}")
        
    except Exception as e:
        print(f"Error predicting house price: {e}")

if __name__ == "__main__":
    main()
