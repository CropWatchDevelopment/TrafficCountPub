import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
desired_fps = config.get('desired_fps', 15)
video_source = config.get('video_source', 'rtsp://username:password@192.168.1.48/axis-media/media.amp')

roi_lines = config.get('roi_lines', [])
supabase_url = config.get('supabase_url', 'YOUR_SUPABASE_URL')
supabase_key = config.get('supabase_key', 'YOUR_SUPABASE_KEY')
dev_eui = config.get('dev_eui', 'YOUR_DEV_EUI')
