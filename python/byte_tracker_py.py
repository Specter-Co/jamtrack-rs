from byte_tracker import ByteTracker, Object, PyRect
from typing import List
import numpy as np
from numpy import ndarray
import pandas as pd

class ByteTrackerPy:
    def __init__(
        self,
        frame_rate: int,
        track_buffer: int,
        track_thresh: float,
        high_thresh: float,
        match_thresh: float
    ):

        # Create a ByteTracker
        self.tracker = ByteTracker(
            frame_rate=frame_rate,
            track_buffer=track_buffer,
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            match_thresh=match_thresh
        )
    
    def update(self, dets: ndarray):
        """
        dets: array of shape n_det x 10
        columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z 
        """
        objects = []
        for det in dets:
            obj = Object(
                detection_id=0,
                rect=PyRect(*det[2:6]),
                track_id=np.uint32(det[1]),
                prob=det[6],
                track_vel_xy=None,
            )
            objects.append(obj)
        tracks = self.tracker.update(objects)
        return tracks
 