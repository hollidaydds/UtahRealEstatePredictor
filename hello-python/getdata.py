import os
import flask
import requests
import json
import pandas as pd
from datetime import datetime
import time

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def get_utah_home_data():
    headers = {
        'accept': 'application/json',
        'X-Api-Key': '699ec3452ff6455899970e158e981e37'
    }
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'utah_sales_{timestamp}.csv'
        total_records = 0
        first_batch = True
        retry_count = 0
        max_retries = 3
        stop_reason = "Unknown"

        # Start from where we left off (55000) and continue to 500000
        for offset in range(55000, 500000, 500):
            try:
                print(f"Fetching records {offset} to {offset + 500}...")
                
                # Add a small delay between requests to avoid rate limiting
                time.sleep(0.5)  # 500ms delay
                
                response = requests.get(
                    'https://api.rentcast.io/v1/listings/sale',
                    headers=headers,
                    params={
                        'limit': 500,
                        'offset': offset,
                        'state': 'UT'
                    }
                )
                
                # Handle rate limiting
                if response.status_code == 429:  # Too Many Requests
                    print(f"Rate limit hit at offset {offset}")
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"Waiting 60 seconds before retry {retry_count}/{max_retries}")
                        time.sleep(60)  # Wait 60 seconds before retrying
                        continue
                    else:
                        stop_reason = f"Rate limit exceeded after {max_retries} retries"
                        break
                
                # Reset retry count on successful request
                retry_count = 0
                
                listings = response.json()
                
                if not listings:  # Stop if no more data
                    stop_reason = f"No more listings found after offset {offset}"
                    print(stop_reason)
                    break
                    
                # Convert batch to DataFrame
                df = pd.DataFrame(listings)
                batch_size = len(df)
                print(f"Retrieved {batch_size} records in this batch")
                
                # Write to CSV (append mode after first batch)
                if first_batch:
                    df.to_csv(csv_filename, index=False)
                    first_batch = False
                else:
                    df.to_csv(csv_filename, mode='a', header=False, index=False)
                
                total_records += batch_size
                
                # Show only the last batch in the web interface
                display_listings = listings
                
            except Exception as batch_error:
                print(f"Error processing batch at offset {offset}: {str(batch_error)}")
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(60)  # Wait 60 seconds before retrying
                    continue
                else:
                    stop_reason = f"Stopped due to repeated errors: {str(batch_error)}"
                    break
        
        return flask.render_template('index.html', 
                                  listings=display_listings,  # Show only last batch
                                  csv_filename=csv_filename,
                                  total_records=total_records,
                                  message=f"Data has been saved to CSV. Retrieved {total_records} records. {stop_reason}. Only showing the last batch in preview.")
    except Exception as e:
        return flask.render_template('index.html', error=str(e))

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return flask.send_file(filename,
                             mimetype='text/csv',
                             as_attachment=True,
                             download_name=filename)
    except Exception as e:
        return str(e), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8111))
    app.run(host='0.0.0.0', port=port)