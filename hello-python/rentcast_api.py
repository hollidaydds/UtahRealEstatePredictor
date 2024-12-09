import requests
import json
import sys
import traceback

RENTCAST_API_KEY = "699ec3452ff6455899970e158e981e37"

def get_home_value(address, property_type="Single Family", bedrooms=None, bathrooms=None, square_footage=None, comp_count=5):
    """
    Get home value estimate from Rentcast API
    
    Args:
        address (str): Full address of the property
    
    Returns:
        dict: API response containing home value estimate and comparables
    """
    try:
        # Base URL for the Rentcast properties endpoint
        base_url = "https://api.rentcast.io/v1/properties"

        # Build query parameters
        params = {
            "address": address,
        }

        # Headers for the API request
        headers = {
            "accept": "application/json",
            "X-Api-Key": RENTCAST_API_KEY  # Use the API key variable
        }
        
        print(f"Making API request to: {base_url}", flush=True)
        print(f"With params: {json.dumps(params, indent=2)}", flush=True)
        print(f"Headers: {json.dumps(headers, indent=2)}", flush=True)
        
        # Make the API request
        response = requests.get(base_url, headers=headers, params=params)
        
        print(f"Response status code: {response.status_code}", flush=True)
        print(f"Full response text: {response.text}", flush=True)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse and return the JSON response
        data = response.json()
        print(f"Parsed JSON data structure: {json.dumps(data, indent=2)}", flush=True)
        
        # Handle case where API returns a list
        if isinstance(data, list) and len(data) > 0:
            print("Response is a list, using first item", flush=True)
            data = data[0]
            
        # Extract and format the relevant data
        formatted_data = {
            'price': data.get('price', 0),
            'priceRangeLow': data.get('price', 0) * 0.9,  # Estimate 10% range
            'priceRangeHigh': data.get('price', 0) * 1.1,
            'bedrooms': data.get('bedrooms', 0),
            'bathrooms': data.get('bathrooms', 0),
            'squareFootage': data.get('squareFootage', 0),
            'lotSize': data.get('lotSize', 0),
            'yearBuilt': data.get('yearBuilt', 0),
            'county': data.get('county', ''),
            'address': data.get('address', ''),
            'city': data.get('city', ''),
            'state': data.get('state', ''),
            'zipCode': data.get('zipCode', ''),
            'comparables': data.get('comparables', [])  # Add comparables
        }
        
        print(f"Formatted data: {json.dumps(formatted_data, indent=2)}", flush=True)
        return formatted_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}", flush=True)
        return {"error": str(e)}
    except Exception as e:
        print(f"Unexpected error: {str(e)}", flush=True)
        traceback.print_exc()
        return {"error": str(e)}

def get_rentcast_estimate(address=None):
    """
    Get property value estimate from Rentcast AVM API
    
    Args:
        address (str): Full address of the property
    
    Returns:
        dict: API response containing property value estimate
    """
    try:
        # Base URL for the Rentcast AVM endpoint
        base_url = "https://api.rentcast.io/v1/avm/value"

        # Build query parameters
        params = {
            "address": address,
        }

        # Headers for the API request
        headers = {
            "accept": "application/json",
            "X-Api-Key": RENTCAST_API_KEY
        }
        
        print(f"Making AVM API request to: {base_url}", flush=True)
        print(f"With params: {json.dumps(params, indent=2)}", flush=True)
        
        # Make the API request
        response = requests.get(base_url, headers=headers, params=params)
        
        print(f"Response status code: {response.status_code}", flush=True)
        print(f"Full response text: {response.text}", flush=True)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse and return the JSON response
        data = response.json()
        print(f"Parsed JSON data structure: {json.dumps(data, indent=2)}", flush=True)
        
        return data

    except Exception as e:
        print(f"Error in get_rentcast_estimate: {str(e)}", flush=True)
        return {"error": str(e)}
