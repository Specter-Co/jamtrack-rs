import json
import glob
import pandas as pd
import cv2
import os
from pathlib import Path
import numpy as np
import re

def get_video_timestamps(std_out_path):
    with open(std_out_path, 'r') as f:
        for line in f:
            if 'segment_timestamps:' in line:
                timestamps_strings = line.split('segment_timestamps:')[1].strip().split('latency:')[0].strip()
                timestamps = [float(timestamp.split('s')[0].replace('[', '')) for timestamp in timestamps_strings.split()]
                print(f"Found {len(timestamps)} timestamps in stdout")
                return np.array(timestamps)
    raise ValueError(f"No timestamps found in {std_out_path}")

def load_all_video_frames(video_path, std_out_path=''):
    if not std_out_path:
        std_out_path = [os.path.join(os.path.dirname(video_path), file) for file in os.listdir(os.path.dirname(video_path)) if file.endswith(".txt")][0]
    timestamps = get_video_timestamps(std_out_path)

    cap = cv2.VideoCapture(video_path)
    n_frames = len(timestamps)
    all_frames = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        all_frames.append(frame)
        frame_idx += 1
    print(f"Loaded {len(all_frames)} frames from {video_path}")
    return all_frames

def dump_all_video_frames(video_path, std_out_path=''):
    if not std_out_path:
        std_out_path = [os.path.join(os.path.dirname(video_path), file) for file in os.listdir(os.path.dirname(video_path)) if file.endswith(".txt")][0]
    timestamps = get_video_timestamps(std_out_path)

    cap = cv2.VideoCapture(video_path)
    n_frames = len(timestamps)
    frame_idx = 0

    save_dir = Path(video_path)
    save_dir = str(save_dir.parent) + '/' + str(save_dir.stem) + '-imgs'
    print(save_dir)
    if os.path.exists(save_dir):
        print(f'Directory {save_dir} already exists, skipping')
        return sorted(glob.glob(save_dir+'/*.jpg'))
        
    os.makedirs(save_dir, exist_ok=False)
            
    print(f'processing video {video_path} with {n_frames} frames')
    while cap.isOpened() and frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            continue
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{save_dir}/frame_{frame_idx:05d}.jpg" , frame)
        
        frame_idx += 1

    print(f"\nDone dumping {video_path}")
    return sorted(glob.glob(save_dir+'/*.jpg'))

def align_timestamps_to_frames(timestamps_ms, df):
    df['frame'] = df['timestamp_ms'].apply(lambda x: np.argmin(np.abs(timestamps_ms - x)))
    return df

def convert_qid_dets_to_df(qid_json_path, sensor_id, timestamps_path):
    class_name_to_idx = {
        "Person": 0,
        "Bag": 1,
    }
    # Parse json
    with open(qid_json_path, 'r') as f:
        raw = f.read()
    # Fix: Split the raw text into separate JSON arrays
    arrays = re.findall(r'\[.*?\]', raw, re.DOTALL)
    # Parse each array individually
    det_df = {
        "class": [],
        "confidence": [],
        "det_id": [],
        "sensor_id_hex": [],
        "sensor_id": [],
        "timestamp_s": [],
        "track_info": [],
        "x1_r": [],
        "y1_r": [],
        "x2_r": [],
        "y2_r": [],
    }
    # Unpack detections from json into df
    for array_str in arrays:
        frame = json.loads(array_str)  # Combine all dicts into one list
        for det in frame:
            if det['sensor_id']['id'] != int(sensor_id, 16):
                continue
            det_df['class'].append(det['class'])
            det_df['confidence'].append(det['confidence'])
            det_df['det_id'].append(det['id'])
            det_df['sensor_id_hex'].append(det['sensor_id']['id'])
            det_df['sensor_id'].append(hex(det['sensor_id']['id']))
            det_df['track_info'].append(det['track_info'])
            det_df['x1_r'].append(det['roi']['left'])
            det_df['y1_r'].append(det['roi']['top'])
            det_df['x2_r'].append(det['roi']['right'])
            det_df['y2_r'].append(det['roi']['bottom'])
            ts = det['timestamp']['secs'] + det['timestamp']['nanos'] * 1e-9 
            det_df['timestamp_s'].append(ts)
    det_df = pd.DataFrame(det_df)
    det_df['timestamp_ms'] = det_df['timestamp_s'] * 1000.
    det_df['cls'] = det_df['class'].map(class_name_to_idx)

    # Align to frames
    timestamps = get_video_timestamps(timestamps_path)
    det_df = align_timestamps_to_frames(timestamps * 1000., det_df)
    return det_df