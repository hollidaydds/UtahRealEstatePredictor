import requests
import json
import sys

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
            property_data = data[0]  # Take the first property from the list
        else:
            print("Response is not a list, using as is", flush=True)
            property_data = data
            
        print(f"Property data being used: {json.dumps(property_data, indent=2)}", flush=True)
            
        # Extract property details with proper field names from the API response
        result = {
            "formattedAddress": property_data.get("formattedAddress", ""),
            "bedrooms": property_data.get("bedrooms", 0),
            "bathrooms": property_data.get("bathrooms", 0),
            "squareFootage": property_data.get("squareFootage", 0)
        }
        
        print(f"Final formatted result: {json.dumps(result, indent=2)}", flush=True)
        return result
        
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response:
            try:
                error_msg = f"{error_msg} - Response: {e.response.text}"
                print(f"Response headers: {dict(e.response.headers)}", flush=True)
                print(f"Response status: {e.response.status_code}", flush=True)
            except:
                pass
        print(f"Error making API request: {error_msg}", flush=True)
        return {"error": f"Failed to get property details: {error_msg}"}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}", flush=True)
        return {"error": "Invalid JSON response from API"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}", flush=True)
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
