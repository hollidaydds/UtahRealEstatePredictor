# Utah House Price Prediction Web Application

A Flask-based web application that predicts house prices in Utah using machine learning and provides rental estimates through the Rentcast API. The application features an interactive map interface powered by Google Maps and uses real Utah housing data for accurate predictions.

## Features

- House price prediction using machine learning trained on Utah housing data
- Rental value estimates via Rentcast API
- Interactive Google Maps integration for address selection
- Address autocomplete functionality
- Property details input form
- Real-time price predictions
- Data cleaning and preprocessing utilities
- Synthetic data generation capabilities

## Prerequisites

- Python 3.10 or newer
- pip (Python package installer)
- API keys for:
  - Google Maps API
  - Rentcast API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/utah-house-price-predictor.git
cd hello-python
```

2. Set up API keys:
   - Rename `api_keys_template.py` to `api_keys.py`
   - Add your API keys to `api_keys.py`:
     ```python
     GOOGLE_MAPS_API_KEY = 'your_google_maps_api_key'
     RENTCAST_API_KEY = 'your_rentcast_api_key'
     ```

3. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Use the provided batch file:
```bash
run_app.bat
```

Or manually start the Flask server:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

- `main.py` - Flask application entry point and route definitions
- `predict_house_price.py` - Machine learning model for house price predictions
- `rentcast_api.py` - Rentcast API integration for rental estimates
- `data_cleaning.py` - Data preprocessing and cleaning utilities
- `generate_synthetic.py` - Synthetic data generation for model training
- `price_prediction.py` - Core price prediction logic
- `remove_outliers.py` - Outlier detection and removal
- `templates/index.html` - Main web interface
- `api_keys.py` - API key configuration (not included in repository)
- `requirements.txt` - Python package dependencies
- `run_app.bat` - Batch file for running the application
- `setup_env.bat` - Environment setup script

## Dependencies

Main dependencies include:
- Flask==3.0.3 - Web framework
- numpy==1.23.5 - Numerical computing
- pandas==1.5.3 - Data manipulation
- scikit-learn==1.3.0 - Machine learning
- xgboost - Gradient boosting
- shap - Model interpretability
- matplotlib==3.8.2 - Data visualization
- seaborn==0.13.0 - Statistical visualizations

## Data Files

- `Cleaned.csv` - Processed dataset
- `utah_sales_*.csv` - Raw Utah housing data
- `utah_sales_no_outliers.csv` - Dataset with outliers removed
- `utah_sales_with_synthetic.csv` - Dataset augmented with synthetic data
- `house_price_model.pkl` - Trained machine learning model

## Troubleshooting

1. **Package Installation Issues**
   - Use Python 3.10 for best compatibility
   - Install packages with `pip install --only-binary=:all:` if encountering build errors
   - Ensure all dependencies are installed within the virtual environment

2. **API Integration**
   - Verify API keys are correctly set in `api_keys.py`
   - Check API quotas and limits
   - Ensure proper network connectivity

3. **Model Performance**
   - Verify the presence of `house_price_model.pkl`
   - Check input data format matches training data
   - Review data cleaning parameters if predictions seem incorrect

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- Utah housing data providers
- Rentcast API for rental estimates
- Google Maps Platform for location services
