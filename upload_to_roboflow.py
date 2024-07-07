from roboflow import Roboflow
import logging
from datetime import datetime

def upload_image_to_roboflow(image_path, track_id):
    try:
        rf = Roboflow(api_key="a2815vKEFlXlfC19SXFA")
        workspace_id = 'cropwatch'
        project_id = 'testuploadproject'
        project = rf.workspace(workspace_id).project(project_id)

        current_date = datetime.now().strftime("%Y-%m-%d")
        batch_name = f'loravis-auto-{current_date}'

        project.upload(
            image_path,
            batch_name=batch_name,
            split="train",
            num_retry_uploads=3,
            tag="auto_upload",
            sequence_size=100
        )
        logging.info(f"Image {image_path} uploaded successfully for track ID {track_id}")
    except Exception as e:
        logging.error(f"Failed to upload image {image_path} for track ID {track_id}: {e}")
