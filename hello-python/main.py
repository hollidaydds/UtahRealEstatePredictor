import os
from flask import Flask, render_template, request, jsonify
from rentcast_api import get_home_value, get_rentcast_estimate

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
                             sqft=sqft)
    return render_template('index.html')

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8111))
    app.run(host='0.0.0.0', port=port)
