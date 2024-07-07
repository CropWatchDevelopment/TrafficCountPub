import cv2
import logging
from datetime import datetime, timedelta
from helper import configure_logging
from config import config, roi_lines, desired_fps, video_source, dev_eui
from tracking import create_tracker
from detection import process_frame, process_detections
from database import insert_traffic_data_async

# Configure logging
configure_logging()

# Set the debug mode
debug_mode = True  # Set this flag to True to enable visual display

# Set up video capture
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    logging.error("Error: Could not open video source.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = desired_fps

logging.info(f"Video source opened: {video_source}, Frame width: {frame_width}, Frame height: {frame_height}, Frame rate: {frame_rate}")

# Initialize tracker
tracker = create_tracker(frame_rate)

# Initialize cross counts
cross_counts = {cls: 0 for line in roi_lines for cls in line['classes']}
object_counts = {cls: 0 for cls in cross_counts.keys()}  # Initialize the new variable

# Set up periodic data insertion
insertion_period = timedelta(minutes=1)
next_insertion_time = datetime.utcnow() + insertion_period

def update_count_callback(class_name):
    global object_counts
    object_counts[class_name] += 1

def perform_periodic_insertion():
    global next_insertion_time
    if datetime.utcnow() >= next_insertion_time:
        # Prepare data for Supabase insertion
        data = {
            'people': object_counts.get('person', 0),
            'bicycle': object_counts.get('bicycle', 0),
            'car': object_counts.get('car', 0),
            'truck': object_counts.get('truck', 0),
            'bus': object_counts.get('bus', 0),
            'dev_eui': dev_eui
        }

        logging.info(f"Data prepared for insertion: {data}")

        # Insert data into Supabase asynchronously
        insert_traffic_data_async(data)

        # Log current counts
        logging.info("Current counts before reset:")
        for obj_type, total_count in object_counts.items():
            if total_count > 0:
                logging.info(f"{obj_type}: {total_count}")

        next_insertion_time = datetime.utcnow() + insertion_period

        # Reset counts after insertion
        for key in object_counts.keys():
            object_counts[key] = 0
        logging.info(f"Reset object_counts: {object_counts}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, dets, boxes_scores, img_info, img_size = process_frame(frame)

    # Update the tracker
    online_targets = tracker.update(boxes_scores, img_info, img_size)

    # Process detections and draw results
    frame = process_detections(frame, online_targets, dets.cpu().numpy(), boxes_scores, update_count_callback)

    # Perform periodic insertion
    perform_periodic_insertion()

    # Show the frame with bounding boxes and tracking info
    if debug_mode:
        # Draw ROI lines
        for line in roi_lines:
            start = tuple(line['start'])
            end = tuple(line['end'])
            cv2.line(frame, start, end, (0, 255, 255), 2)
            for cls in line['classes']:
                cv2.putText(frame, cls, (start[0] + 5, start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display counts
        person_count = object_counts.get('person', 0)
        vehicle_count = sum(object_counts.get(cls, 0) for cls in ['car', 'truck', 'bus', 'bicycle', 'motorcycle'])
        count_text = f"Person: {person_count}, Vehicles: {vehicle_count}"
        
        text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = frame_width - text_size[0] - 10
        text_y = frame_height - 10

        cv2.putText(frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
