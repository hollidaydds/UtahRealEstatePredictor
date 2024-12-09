import requests
import json

def get_home_value(address, property_type="Single Family", bedrooms=None, bathrooms=None, square_footage=None, comp_count=5):
    """
    Get home value estimate from Rentcast API
    
    Args:
        address (str): Full address of the property
    
    Returns:
        dict: API response containing home value estimate and comparables
    """
    try:
        # Base URL for the Rentcast AVM endpoint
        base_url = "https://api.rentcast.io/v1/properties"

        # Build query parameters
        params = {
            "address": address,
        }

        # Headers for the API request
        headers = {
            "accept": "application/json",
            "X-Api-Key": "699ec3452ff6455899970e158e981e37"  # Your Rentcast API key
        }
        
        print(f"Making API request to: {base_url}")
        print(f"With params: {json.dumps(params, indent=2)}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        
        # Make the API request
        response = requests.get(base_url, headers=headers, params=params)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text[:500]}...")  # Print first 500 chars of response
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse and return the JSON response
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Error response: {e.response.text}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}")
        return {"error": "Invalid JSON response from API"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"error": str(e)}
