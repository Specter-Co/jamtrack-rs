from byte_tracker_py import ByteTrackerPy, Object
from typing import List
import numpy as np
from numpy import ndarray
import pandas as pd

def load_mot_data(file_path: str) -> pd.DataFrame:
    # Define the dtype dictionary for your columns
    names = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    return pd.read_csv(file_path, names=names)

# Testing
f = "/home/keller/Documents/specter-work/MOT17/train/MOT17-02-DPM/det/det.txt"
df_det = load_mot_data(f)

bytetracker = ByteTrackerPy(
    frame_rate=10,
    track_buffer=5,
    track_thresh=0.5,
    high_thresh=0.7,
    match_thresh=0.7
)
print(df_det)

# Run tracker over detections
columns = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "vel_x", "vel_y"]
tracker_matrix = np.empty([0,len(columns)])
for frame_index, dets in df_det.groupby('frame'):
    outputs: List[Object] = bytetracker.update(dets.values)
    for output in outputs:
        data = np.array([
            frame_index, 
            output.track_id, 
            output.rect.x, 
            output.rect.y, 
            output.rect.width, 
            output.rect.height, 
            output.track_vel_xy[0], 
            output.track_vel_xy[1]
        ])
        tracker_matrix = np.vstack([tracker_matrix, data])

# Collect results of tracker 
df_tracker = pd.DataFrame(data=tracker_matrix, columns=columns)

# # Visualize results
# Iterate over frames
for frame_index, dets in df_det.groupby('frame'):
    tracker_matrix = df_tracker[df_tracker["frame"] == frame_index]

# Draw raw detections, draw tracker boxes with track IDs, draw velocity vector

# Save as mp4
