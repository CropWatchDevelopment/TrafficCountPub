import sys
from collections import namedtuple

# Add ByteTrack directory to sys.path
sys.path.append('./ByteTrack')

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

TrackerArgs = namedtuple('TrackerArgs', ['track_thresh', 'track_buffer', 'match_thresh', 'min_box_area', 'mot20'])

def create_tracker(frame_rate, track_thresh=0.5, track_buffer=70, match_thresh=0.7, min_box_area=5, mot20=False):
    args = TrackerArgs(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        min_box_area=min_box_area,
        mot20=mot20
    )
    return BYTETracker(args=args, frame_rate=frame_rate)
