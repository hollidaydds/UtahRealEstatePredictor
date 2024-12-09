from flask import Flask, render_template, request, jsonify
import os
import json
import traceback
from predict_house_price import predict_house_price
from rentcast_api import get_home_value

try:
    from secrets import GOOGLE_MAPS_API_KEY
except ImportError:
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in secrets.py or environment variables")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        address = request.form.get('address')
        zipcode = request.form.get('zipcode')
        bedrooms = request.form.get('bedrooms')
        bathrooms = request.form.get('bathrooms')
        sqft = request.form.get('sqft')
        return render_template('index.html', 
                             address=address,
                             zipcode=zipcode, 
                             bedrooms=bedrooms,
                             bathrooms=bathrooms,
                             sqft=sqft,
                             google_maps_api_key=GOOGLE_MAPS_API_KEY)
    return render_template('index.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.route('/get_rentcast_estimate')
def get_rentcast_estimate_route():
    address = request.args.get('address')
    if not address:
        return jsonify({"error": "Address is required"}), 400
        
    try:
        estimate_data = get_home_value(address)
        return jsonify(estimate_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_estimate')
def get_estimate():
    address = request.args.get('address')
    if not address:
        return jsonify({"error": "Address is required"}), 400
        
    try:
        print(f"Calling get_rentcast_estimate with address: {address}", flush=True)
        estimate_data = get_rentcast_estimate(address)
        print(f"Got estimate data: {estimate_data}", flush=True)
        return jsonify(estimate_data)
    except Exception as e:
        import traceback
        print(f"Error in get_estimate: {str(e)}", flush=True)
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route('/rentcast', methods=['POST'])
def get_rentcast():
    try:
        print("\n=== Starting /rentcast endpoint ===", flush=True)
        data = request.get_json()
        address = data.get('address')
        print(f"Received data from frontend: {json.dumps(data, indent=2)}", flush=True)
        
        if not address:
            print("No address provided", flush=True)
            return jsonify({"error": "Address is required"}), 400
            
        print("Calling get_home_value...", flush=True)
        result = get_home_value(address)
        print(f"Raw Rentcast API result: {json.dumps(result, indent=2)}", flush=True)
        
        if "error" in result:
            print(f"Error from Rentcast API: {result['error']}", flush=True)
            return jsonify({"error": result["error"]}), 400
            
        # Use the updated form data for prediction if available, otherwise use Rentcast data
        prediction_data = {
            'bedrooms': float(data.get('bedrooms', result.get('bedrooms', 0))),
            'bathrooms': float(data.get('bathrooms', result.get('bathrooms', 0))),
            'square_footage': float(data.get('square_footage', result.get('squareFootage', 0))),
            'lot_size': float(data.get('lot_size', result.get('lotSize', 0) / 43560)),  # Convert to acres if using Rentcast data
            'year_built': int(data.get('year_built', result.get('yearBuilt', 0))),
            'county': data.get('county', result.get('county', '')),
            'property_type': 'Single Family'
        }
        print(f"Prepared prediction data: {json.dumps(prediction_data, indent=2)}", flush=True)
        
        # Get our prediction with the updated values
        print("Calling predict_house_price...", flush=True)
        predicted_price = predict_house_price(
            bedrooms=prediction_data['bedrooms'],
            bathrooms=prediction_data['bathrooms'],
            square_footage=prediction_data['square_footage'],
            lot_size=prediction_data['lot_size'],
            year_built=prediction_data['year_built'],
            county=prediction_data['county'],
            property_type=prediction_data['property_type']
        )
        print(f"Got prediction: ${predicted_price:,.2f}", flush=True)
        
        # Keep all the original Rentcast data and add our prediction
        response_data = {
            **result,  # Include all original Rentcast data
            'predicted_price': predicted_price  # Add our prediction
        }
        
        print(f"Final response being sent to frontend: {json.dumps(response_data, indent=2)}", flush=True)
        print("=== Finished /rentcast endpoint ===\n", flush=True)
            
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_rentcast: {str(e)}", flush=True)
        print("Full traceback:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])

def predict():
    try:
        data = request.get_json()
        predicted_price = predict_house_price(
            bedrooms=data['bedrooms'],
            bathrooms=data['bathrooms'],
            square_footage=data['square_footage'],
            lot_size=data['lot_size'],
            year_built=data['year_built'],
            county=data['county'],
            property_type=data['property_type']
        )
        return jsonify({'predicted_price': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8111))
    app.run(host='0.0.0.0', port=port)
