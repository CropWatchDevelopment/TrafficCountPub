import torch
import cv2
import numpy as np
from config import roi_lines, desired_fps
from helper import box_intersects_line
from upload_to_roboflow import upload_image_to_roboflow
import threading
import datetime

# Initialize the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Load YOLOv5m

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class names and allowed classes
class_names = model.names
allowed_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']
allowed_class_ids = [k for k, v in class_names.items() if v in allowed_classes]

cross_counts = {cls: 0 for cls in allowed_classes}
counted_ids = {cls: set() for cls in allowed_classes}

last_known_positions = {}
recently_counted = {}  # Dictionary to keep track of recently counted objects

debounce_time = 3  # Debounce time in seconds to avoid double counts

debug_mode = True  # Set this flag to True to enable visual display

def async_upload(image_path, track_id):
    upload_thread = threading.Thread(target=upload_image_to_roboflow, args=(image_path, track_id))
    upload_thread.start()

def process_frame(frame, frame_size=(640, 480)):
    # Preprocess the frame for inference
    frame = cv2.resize(frame, frame_size)

    # Perform object detection
    results = model(frame)
    detection_results = results.xyxy[0].cpu().numpy()
    filtered_detections = [result for result in detection_results if result[4] > 0.5 and int(result[5]) in allowed_class_ids]

    detections = [[x1, y1, x2, y2, conf, cls] for x1, y1, x2, y2, conf, cls in filtered_detections]

    if detections:
        dets = torch.tensor(detections).to(device)
    else:
        dets = torch.empty((0, 6)).to(device)

    img_info = frame.shape[:2]
    img_size = [frame.shape[0], frame.shape[1]]

    post_boxes = np.array([result[:4] for result in filtered_detections])  # Ensure only bbox coordinates
    scores = np.array([result[4] for result in filtered_detections])
    
    boxes_scores = np.hstack((post_boxes, scores.reshape(-1, 1))) if post_boxes.size > 0 else np.empty((0, 5))

    return frame, dets, boxes_scores, img_info, img_size

def process_detections(frame, online_targets, detections, post_boxes, update_count_callback=None):
    global last_known_positions, cross_counts, counted_ids, recently_counted

    current_time = datetime.datetime.utcnow()

    for an_online_target in online_targets:
        track_xyxy = [an_online_target.tlwh[0], an_online_target.tlwh[1], an_online_target.tlwh[0] + an_online_target.tlwh[2], an_online_target.tlwh[1] + an_online_target.tlwh[3]]
        det_i = next((i for i, a_box in enumerate(post_boxes) if np.allclose(track_xyxy, a_box[:4], rtol=1.0, atol=1.0)), None)

        if det_i is not None:
            cl = int(detections[det_i][5])
            class_name = class_names[cl]
            if class_name not in allowed_classes:
                continue

            x1, y1, w, h = an_online_target.tlwh
            x2, y2 = x1 + w, y1 + h

            last_known_positions[an_online_target.track_id] = {'bbox': (x1, y1, x2, y2), 'class_name': class_name}

            if debug_mode:
                # Draw bounding box with different colors for people and vehicles
                color = (255, 0, 0) if class_name == 'person' else (0, 0, 255)  # Blue for people, Red for vehicles
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # Add tracking ID and confidence
                cv2.putText(frame, f'ID: {an_online_target.track_id} Conf: {detections[det_i][4]:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for line in roi_lines:
                if class_name in line['classes']:
                    line_start = line['start']
                    line_end = line['end']
                    if box_intersects_line(x1, y1, x2, y2, line_start, line_end):
                        # Check debounce logic
                        if an_online_target.track_id not in recently_counted or (current_time - recently_counted[an_online_target.track_id]).total_seconds() > debounce_time:
                            cross_counts[class_name] += 1
                            counted_ids[class_name].add(an_online_target.track_id)
                            recently_counted[an_online_target.track_id] = current_time
                            if update_count_callback:
                                update_count_callback(class_name)

                            if class_name == 'person':
                                person_image = frame[int(y1):int(y2), int(x1):int(x2)]
                                a = datetime.datetime.now()
                                image_path = f'person_{"%s:%s.%s" % (a.minute, a.second, str(a.microsecond)[:2])}.jpg'
                                cv2.imwrite(image_path, person_image)
                                async_upload(image_path, an_online_target.track_id)
        else:
            if an_online_target.track_id in last_known_positions:
                position_info = last_known_positions[an_online_target.track_id]
                x1, y1, x2, y2 = position_info['bbox']
                class_name = position_info['class_name']
                if debug_mode and class_name:
                    # Draw bounding box with different colors for people and vehicles
                    color = (255, 0, 0) if class_name == 'person' else (0, 0, 255)  # Blue for people, Red for vehicles
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # Add tracking ID
                    cv2.putText(frame, f'ID: {an_online_target.track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Remove old entries from recently_counted
    recently_counted = {track_id: time for track_id, time in recently_counted.items() if (current_time - time).total_seconds() <= debounce_time}

    return frame
