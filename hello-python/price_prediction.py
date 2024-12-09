import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import pickle
import os

def validate_real_estate_data(X, y):
    """Validate and clean real estate data"""
    # Create masks for each validation criteria
    price_mask = (y >= 10000) & (y <= 10000000)  # Price between $10k and $10M
    bed_mask = (X['bedrooms'] >= 1) & (X['bedrooms'] <= 10)
    bath_mask = (X['bathrooms'] >= 1) & (X['bathrooms'] <= 10)
    sqft_mask = (X['squareFootage'] >= 500) & (X['squareFootage'] <= 15000)
    
    # Combine all masks
    valid_mask = (
        price_mask & 
        bed_mask & 
        bath_mask & 
        sqft_mask
    )
    
    # Print validation summary
    total_rows = len(X)
    valid_rows = valid_mask.sum()
    print("\nData Validation Summary:")
    print(f"Total rows: {total_rows}")
    print(f"Valid rows: {valid_rows}")
    print(f"Removed rows: {total_rows - valid_rows}")
    
    return X[valid_mask], y[valid_mask]

def engineer_features(X):
    """Engineer new features to improve model performance"""
    X_new = X.copy()
    
    # Square footage features
    X_new['log_sqft'] = np.log1p(X_new['squareFootage'])
    X_new['sqft_per_room'] = X_new['squareFootage'] / (X_new['bedrooms'] + X_new['bathrooms'])
    
    # Room features
    X_new['total_rooms'] = X_new['bedrooms'] + X_new['bathrooms']
    X_new['bath_per_bed'] = X_new['bathrooms'] / X_new['bedrooms']
    
    # One-hot encode ZIP codes
    X_new['zipCode'] = X_new['zipCode'].astype(str).str.zfill(5)
    zip_dummies = pd.get_dummies(X_new['zipCode'], prefix='zip')
    X_new = pd.concat([X_new, zip_dummies], axis=1)
    
    # Drop original columns
    X_new = X_new.drop(['zipCode'], axis=1)
    
    return X_new

def load_and_preprocess_data(file_path):
    """Load and preprocess the Utah sales data"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select core features
    features = ['bedrooms', 'bathrooms', 'squareFootage', 'zipCode']
    target = 'price'
    
    # Create feature matrix X and target vector y
    X = df[features].copy()
    y = df[target].copy()
    
    # Convert data types
    for col in X.columns:
        if col != 'zipCode':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Handle missing values
    numeric_columns = ['bedrooms', 'bathrooms', 'squareFootage']
    for column in numeric_columns:
        X[column] = X[column].fillna(X[column].median())
    X['zipCode'] = X['zipCode'].fillna(X['zipCode'].mode()[0])
    
    # Remove rows where target is NaN
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Validate and clean the data
    X, y = validate_real_estate_data(X, y)
    
    # Engineer features
    X = engineer_features(X)
    
    print("\nFinal Dataset Shape:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nNumber of ZIP codes:", sum(1 for col in X.columns if col.startswith('zip_')))
    
    return X, y

def prepare_data(df):
    """Prepare data for modeling"""
    # Numeric features
    numeric_features = ['bedrooms', 'bathrooms', 'squareFootage', 'lotSize', 'yearBuilt']
    
    # Categorical features
    categorical_features = ['county', 'propertyType']
    
    # Prepare X and y
    X = df[numeric_features + categorical_features]
    y = df['price']
    
    # Handle missing values
    if X.isnull().any().any() or y.isnull().any():
        print("Handling missing values...")
        # Impute missing values for numeric features
        for col in numeric_features:
            X[col].fillna(X[col].median(), inplace=True)
        # Impute missing values for categorical features
        for col in categorical_features:
            X[col].fillna(X[col].mode()[0], inplace=True)
        # Impute missing values in target variable
        y.fillna(y.median(), inplace=True)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create full pipeline with best model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=1
        ))
    ])
    
    return X, y, model

def train_model(X, y, model):
    """Train model with optimal parameters"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    # Get feature names after preprocessing
    numeric_features = ['bedrooms', 'bathrooms', 'squareFootage', 'lotSize', 'yearBuilt']
    categorical_features = ['county', 'propertyType']
    
    # Get feature names after preprocessing
    feature_names = (
        numeric_features +
        model.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_features).tolist()
    )
    
    # Get feature importance
    importance = model.named_steps['model'].feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    return model, r2, rmse

def save_model(model, file_path):
    """Save trained model to disk"""
    print(f"\nSaving model to {file_path}...")
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

def load_model(file_path):
    """Load trained model from disk"""
    print(f"\nLoading model from {file_path}...")
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model

def predict_price(model, data):
    """Make predictions using the trained model"""
    # Expected features in correct order
    numeric_features = ['bedrooms', 'bathrooms', 'squareFootage', 'lotSize', 'yearBuilt']
    categorical_features = ['county', 'propertyType']
    
    # Convert single sample to DataFrame if it's a dictionary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Ensure all required features are present
    required_features = numeric_features + categorical_features
    missing_features = [col for col in required_features if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Handle missing values
    for col in numeric_features:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)
    for col in categorical_features:
        if data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Make prediction
    prediction = model.predict(data[required_features])
    
    return prediction

def example_prediction():
    """Example of how to use the saved model for prediction"""
    # Load the saved model
    model_file = Path(__file__).parent / 'house_price_model.pkl'
    if not model_file.exists():
        print(f"Error: Model file not found at {model_file}")
        return
    
    model = load_model(model_file)
    
    # Example house data
    example_house = {
        'bedrooms': 4,
        'bathrooms': 2.5,
        'squareFootage': 2500,
        'lotSize': 0.25,
        'yearBuilt': 2000,
        'county': 'Salt Lake',
        'propertyType': 'Single Family'
    }
    
    # Make prediction
    try:
        predicted_price = predict_price(model, example_house)
        print(f"\nPredicted Price: ${predicted_price[0]:,.2f}")
        
        # Multiple predictions example
        example_houses = pd.DataFrame([
            {
                'bedrooms': 3,
                'bathrooms': 2,
                'squareFootage': 1800,
                'lotSize': 0.15,
                'yearBuilt': 1990,
                'county': 'Utah',
                'propertyType': 'Single Family'
            },
            {
                'bedrooms': 5,
                'bathrooms': 3.5,
                'squareFootage': 3500,
                'lotSize': 0.4,
                'yearBuilt': 2010,
                'county': 'Salt Lake',
                'propertyType': 'Single Family'
            }
        ])
        
        predicted_prices = predict_price(model, example_houses)
        print("\nMultiple Predictions:")
        for i, price in enumerate(predicted_prices, 1):
            print(f"House {i}: ${price:,.2f}")
            
    except Exception as e:
        print(f"Error making prediction: {e}")

def main():
    # Set up file paths using Path
    base_dir = Path(__file__).parent
    data_file = base_dir / 'cleaned.csv'
    model_file = base_dir / 'house_price_model.pkl'
    
    # Verify file exists
    if not data_file.exists():
        print(f"Error: File not found at {data_file}")
        print(f"Current directory: {base_dir}")
        print(f"Available files:")
        for f in base_dir.glob('*.csv'):
            print(f"  - {f.name}")
        return
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(data_file)
    
    # Prepare data and get model
    X, y, model = prepare_data(df)
    
    # Train model and get results
    model, r2, rmse = train_model(X, y, model)
    
    # Save the trained model
    save_model(model, model_file)
    
    # Test loading the model
    loaded_model = load_model(model_file)
    
    # Verify the loaded model works
    print("\nVerifying loaded model...")
    X_test = X.iloc[:5]  # Take first 5 samples as a test
    original_predictions = model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)
    
    # Check if predictions match
    predictions_match = np.allclose(original_predictions, loaded_predictions)
    print(f"Loaded model predictions match original: {predictions_match}")
    
    # Run example prediction
    print("\nTesting model with example predictions...")
    example_prediction()

    print("\nModel training and saving complete!")

if __name__ == "__main__":
    main()
