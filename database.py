import logging
from supabase import create_client, Client
import threading
from datetime import datetime, timedelta
from config import supabase_url, supabase_key

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

# Track the last insertion time
last_insertion_time = None

def insert_or_update_traffic_data(data):
    global last_insertion_time
    current_time = datetime.utcnow()
    current_hour = current_time.replace(minute=0, second=0, microsecond=0).isoformat()

    try:
        # Query the existing record for the current hour
        existing_record_query = supabase.table('cw_traffic2').select('*').eq('created_at', current_hour).eq('dev_eui', data.get('dev_eui')).execute()
        
        if existing_record_query.data:
            # Existing record found, increment the counts
            existing_record = existing_record_query.data[0]
            updated_data = {
                'people_count': existing_record['people_count'] + data.get('people', 0),
                'bicycle_count': existing_record['bicycle_count'] + data.get('bicycle', 0),
                'car_count': existing_record['car_count'] + data.get('car', 0),
                'truck_count': existing_record['truck_count'] + data.get('truck', 0),
                'bus_count': existing_record['bus_count'] + data.get('bus', 0)
            }
            logging.info(f"Updating record ID {existing_record['id']} with data: {updated_data}")
            supabase.table('cw_traffic2').update(updated_data).eq('id', existing_record['id']).execute()
        else:
            # No existing record, insert a new one
            new_record = {
                'created_at': current_hour,
                'people_count': data.get('people', 0),
                'bicycle_count': data.get('bicycle', 0),
                'car_count': data.get('car', 0),
                'truck_count': data.get('truck', 0),
                'bus_count': data.get('bus', 0),
                'dev_eui': data.get('dev_eui')
            }
            logging.info(f"Inserting new record: {new_record}")
            supabase.table('cw_traffic2').insert(new_record).execute()
            last_insertion_time = current_hour

    except Exception as e:
        logging.error(f"Error in insert_or_update_traffic_data: {e}")

def insert_traffic_data_async(data):
    thread = threading.Thread(target=insert_or_update_traffic_data, args=(data,))
    thread.start()
