from byte_tracker import ByteTracker, Object, PyRect
from typing import List
import numpy as np
from numpy import ndarray
import pandas as pd

class ByteTrackerPy:
    def __init__(
        self,
        frame_rate: int = 30,
        track_buffer: int = 30,
        track_thresh: float = .5,
        high_thresh: float = .7,
        match_thresh: float = .7,
        low_conf_match_thresh: float = .5,
        kalman_std_weight_pos: float = 1. / 20.,
        kalman_std_weight_vel: float = 1. / 160.,
        kalman_std_weight_position_meas: float = 1. /20.,
        kalman_std_weight_position_mot: float = 1. / 20.,
        kalman_std_weight_velocity_mot: float = 1. / 160.,
        kalman_std_aspect_ratio_init: float = 1e-2,
        kalman_std_d_aspect_ratio_init: float = 1e-5,
        kalman_std_aspect_ratio_mot: float = 1e-2,
        kalman_std_d_aspect_ratio_mot: float = 1e-5,
        kalman_std_aspect_ratio_meas: float = 1e-1,
    ):

        # Create a ByteTracker
        self.tracker = ByteTracker(
            frame_rate=frame_rate,
            track_buffer=track_buffer,
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            match_thresh=match_thresh,
            low_conf_match_thresh=low_conf_match_thresh,
            kalman_std_weight_pos=kalman_std_weight_pos,
            kalman_std_weight_vel=kalman_std_weight_vel,
            kalman_std_weight_position_meas=kalman_std_weight_position_meas,
            kalman_std_weight_position_mot=kalman_std_weight_position_mot,
            kalman_std_weight_velocity_mot=kalman_std_weight_velocity_mot,
            kalman_std_aspect_ratio_init=kalman_std_aspect_ratio_init,
            kalman_std_d_aspect_ratio_init=kalman_std_d_aspect_ratio_init,
            kalman_std_aspect_ratio_mot=kalman_std_aspect_ratio_mot,
            kalman_std_d_aspect_ratio_mot=kalman_std_d_aspect_ratio_mot,
            kalman_std_aspect_ratio_meas=kalman_std_aspect_ratio_meas,
        )
    
    def update(self, dets: ndarray):
        """
        dets: array of shape n_det x 10
        columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, cls
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
 