import os
from flask import Flask, render_template, request, jsonify
from rentcast_api import get_home_value

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
def get_rentcast_estimate():
    address = request.args.get('address')
    if not address:
        return jsonify({"error": "Address is required"}), 400
        
    try:
        estimate_data = get_home_value(address=address)
        return jsonify(estimate_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8111))
    app.run(host='0.0.0.0', port=port)
