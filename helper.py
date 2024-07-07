import numpy as np
import logging

# Workaround for deprecated np.float
if not hasattr(np, 'float'):
    np.float = np.float64

# Conversion function
def xyxy2tlwh(bbox_xyxy):
    bbox_tlwh = np.empty(4)
    bbox_tlwh[0] = bbox_xyxy[0]  # x1
    bbox_tlwh[1] = bbox_xyxy[1]  # y1
    bbox_tlwh[2] = bbox_xyxy[2] - bbox_xyxy[0]  # width = x2 - x1
    bbox_tlwh[3] = bbox_xyxy[3] - bbox_xyxy[1]  # height = y2 - y1
    return bbox_tlwh

# Check if bounding box intersects with vertical line
def box_intersects_line(x1, y1, x2, y2, line_x):
    return x1 < line_x < x2

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np

def box_intersects_line(x1, y1, x2, y2, line_start, line_end):
    line_x1, line_y1 = line_start
    line_x2, line_y2 = line_end

    if line_y1 == line_y2:  # Horizontal line
        return y1 < line_y1 < y2 and min(x1, x2) < max(line_x1, line_x2) and max(x1, x2) > min(line_x1, line_x2)
    elif line_x1 == line_x2:  # Vertical line
        return x1 < line_x1 < x2 and min(y1, y2) < max(line_y1, line_y2) and max(y1, y2) > min(line_y1, line_y2)
    else:
        # Handle non-vertical/horizontal lines if needed
        return False
