from byte_tracker_py import ByteTrackerPy, Object
import cv2
import os
import json
import re
from itertools import groupby
import numpy as np
from drawing_tools import *
import torchvision
import torch
import math
import subprocess
import argparse
from alert_geometry import QuadAlertGeometry
"""
python python/resim_tracker.py --det_path ./data/specter-data-6-2/test-2/1748907694102.json --video_path ./data/specter-data-6-2/test-2/1748907554795_1748907565999.h265 --out_video ./tracker_result_videos/test-2_tracks.mp4 --start_frame 1000 --end_frame 1100 --sensor_id 120
"""
def parse_sensor_detections(detection_path, video_path, sensor_id=120, expected_frames=None):
    # Try to find stdout path
    files = os.listdir(os.path.dirname(video_path))
    std_out_path = None
    for file in files:
        if file.endswith('.txt'):
            std_out_path = os.path.join(os.path.dirname(video_path), file)
            break
    if std_out_path is not None and os.path.exists(std_out_path):
        with open(std_out_path, 'r') as f:
            for line in f:
                if 'segment_timestamps:' in line:
                    timestamps_strings = line.split('segment_timestamps:')[1].strip().split('latency:')[0].strip()
                    timestamps = [float(timestamp.split('s')[0].replace('[', '')) for timestamp in timestamps_strings.split()]
                    print(f"Found {len(timestamps)} timestamps in stdout")
        missed_frames = []
    else:
        ts_start, ts_end = os.path.basename(video_path).split('_')[:2]
        ts_start, ts_end = float(ts_start), float(ts_end.split('.')[0])
        expected_time = (ts_end - ts_start) * 0.001
        print(f"Expected video length: {expected_time} seconds")
        vid_reader = cv2.VideoCapture(video_path)
        fps = vid_reader.get(cv2.CAP_PROP_FPS)
        expected_frame_count = int(expected_time * fps) if expected_frames is None else expected_frames
        frame_count = 0
        missed_frames = []
        while frame_count < expected_frame_count:
            ret, frame = vid_reader.read()
            if not ret:
                print(f"No frame at {frame_count}")
                missed_frames.append(frame_count)
                frame_count += 1
                continue
            frame_count += 1
        timestamps = np.linspace(ts_start, ts_end, frame_count - len(missed_frames))
        print(f"Found {len(timestamps)} timestamps in video with {len(missed_frames)} missed frames")
                
    with open(detection_path, 'r') as f:
        raw = f.read()

    # Fix: Split the raw text into separate JSON arrays
    arrays = re.findall(r'\[.*?\]', raw, re.DOTALL)
    # Parse each array individually
    all_dets = []
    for array_str in arrays:
        frame = json.loads(array_str)  # Combine all dicts into one list
        for det in frame:
            if det['sensor_id']['id'] != sensor_id:
                continue
            all_dets.append(det)

    print(f"Found {len(all_dets)} detections for sensor {sensor_id}")

    def get_timestamp_key(detection):
        ts = detection['timestamp']
        return ts['secs'] + ts['nanos'] * 1e-9 
    all_dets.sort(key=get_timestamp_key)
    frame_dets = [list(group) for _, group in groupby(all_dets, key=get_timestamp_key)]
    print(f"Found {len(frame_dets)} unique timestamps")
    det_timestamps = [get_timestamp_key(frame[0]) for frame in frame_dets]

    # TODO remove this
    # timestamps = np.linspace(det_timestamps[0], det_timestamps[-1], frame_count)
    return frame_dets, det_timestamps, timestamps, missed_frames

def align_detection_to_video(det_timestamps, vid_timestamps, thresh=.01):
    """
    Align detection timestamps to video timestamps
    """
    assert len(vid_timestamps) >= len(det_timestamps)
    assert vid_timestamps[0] <= det_timestamps[0]
    det_idx = 0
    output = []
    for i, vid_ts in enumerate(vid_timestamps):
        if det_idx >= len(det_timestamps):
            # print(f"No det for frame {i}")
            output.append(None)
            continue
            
        det_ts = det_timestamps[det_idx]
        if vid_ts + thresh < det_ts:
            print(f"TS mismatch {det_ts - vid_ts} from det_ts and vid_ts, skipping frame {i}")
            output.append(None)
            continue
        else:
            output.append(det_idx)
            det_idx += 1
    return output

def postprocess_boxes(boxes, min_conf_thresh_bag=.1, min_conf_thresh_human=.1, nms_iou_bag=.7, nms_iou_human_bag=.9):
    if len(boxes) == 0:
        return np.array([])
    
    boxes = torch.tensor(boxes)
    # x,y,w,h -> x,y,x,y
    boxes[:, 4] = boxes[:, 2] + boxes[:, 4] 
    boxes[:, 5] = boxes[:, 3] + boxes[:, 5] 

    human_boxes = boxes[(boxes[:, -1] == 0) & (boxes[:, -2] > min_conf_thresh_human)]
    bag_boxes = boxes[(boxes[:, -1] == 1) & (boxes[:, -2] > min_conf_thresh_bag)]
    
    # Intra class NMS
    keep_bag_boxes = torchvision.ops.nms(bag_boxes[:, 2:6], bag_boxes[:, -2], nms_iou_bag)
    remain_bag_boxes = bag_boxes[keep_bag_boxes]

    # Cross class NMS
    all_boxes = torch.cat((human_boxes, remain_bag_boxes))
    # Artificially inflate the scores of human boxes to make sure they are not suppressed by bag boxes
    modified_boxes = all_boxes.clone()
    modified_boxes[modified_boxes[:, -1] == 0, -2] += 1.0
    keep_boxes = torchvision.ops.nms(modified_boxes[:, 2:6], modified_boxes[:, -2], nms_iou_human_bag)
    remain_boxes = all_boxes[keep_boxes]

    # Convert back to x,y,w,h
    remain_boxes[:, 4] = remain_boxes[:, 4] - remain_boxes[:, 2]
    remain_boxes[:, 5] = remain_boxes[:, 5] - remain_boxes[:, 3]

    return remain_boxes.numpy()

def calc_ciou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1

    center_dist_sq = ((ax1 + aw / 2.0) - (bx1 + bw / 2.0))**2 + ((ay1 + ah / 2.0) - (by1 + bh / 2.0))**2

    enclose_x1 = min(ax1, bx1)
    enclose_y1 = min(ay1, by1)
    enclose_x2 = max(ax2, bx2)
    enclose_y2 = max(ay2, by2)

    enclose_diag_sq = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2

    v = (4.0 / (math.pi ** 2)) * (math.atan2(aw, ah) - math.atan2(bw, bh)) ** 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area = aw * ah
    b_area = bw * bh
    union_area = a_area + b_area - inter_area + 1e-7
    enclose_area = max(0, enclose_x2 - enclose_x1) * max(0, enclose_y2 - enclose_y1)

    giou = (enclose_area - union_area) / enclose_area

    return center_dist_sq / (enclose_diag_sq + 1e-7), v, giou

def calc_ious(tracks, dets, calc_ciou=False):
    if len(tracks) == 0 or len(dets) == 0:
        return []

    track_boxes = torch.tensor([[track.x, track.y, track.width, track.height] for track in tracks])
    # x,y,w,h -> x,y,x,y
    track_boxes[:, 2] = track_boxes[:, 0] + track_boxes[:, 2] 
    track_boxes[:, 3] = track_boxes[:, 1] + track_boxes[:, 3] 

    det_boxes = torch.tensor(dets)
    # x,y,w,h -> x,y,x,y
    det_boxes[:, 4] = det_boxes[:, 2] + det_boxes[:, 4] 
    det_boxes[:, 5] = det_boxes[:, 3] + det_boxes[:, 5]

    ious = torchvision.ops.box_iou(track_boxes, det_boxes[:, 2:6])

    if calc_ciou:
        cious = torch.zeros_like(ious)
        for track_i, track in enumerate(tracks):
            ax1, ay1, = track.x, track.y
            aw, ah = track.width, track.height
            ax2, ay2 = ax1 + aw, ay1 + ah
            for det_i, det in enumerate(det_boxes[:, 2:6]):
                bx1, by1, bx2, by2 = det[0], det[1], det[2], det[3]
                bw, bh = bx2 - bx1, by2 - by1

                iou = ious[track_i][det_i]
                center_dist_sq = ((ax1 + aw / 2.0) - (bx1 + bw / 2.0))**2 + ((ay1 + ah / 2.0) - (by1 + bh / 2.0))**2

                enclose_x1 = min(ax1, bx1)
                enclose_y1 = min(ay1, by1)
                enclose_x2 = max(ax2, bx2)
                enclose_y2 = max(ay2, by2)

                enclose_diag_sq = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2

                v = (4.0 / (math.pi ** 2)) * (math.atan2(aw, ah) - math.atan2(bw, bh)) ** 2
                alpha = v / (1.0 - iou + v) if iou > 0.0 else 0.0

                ciou = iou - center_dist_sq / (enclose_diag_sq + 1e-7) - alpha * v
                cious[track_i][det_i] = (max(ciou, -1.0) + 1.0) / 2.0
        return cious
    return ious

def extract_track_info(track, track_class):
    x1,y1 = track.x, track.y
    x2,y2 = x1 + track.width, y1 + track.height
    vx,vy = track.track_vel_xy
    return [
        x1,y1,x2,y2,vx,vy,
        track_class, track.track_id, track.detection_id, track.prob
    ]

def draw_debug_detection(frame, frame_no, track_id, tracks, dets, ious, iou_tracks):
    debug_track = None
    for track in tracks:
        if track.track_id == track_id:
            debug_track = track
            break
    if debug_track is None:
        return

    det_id = debug_track.detection_id
    debug_det = None
    det_idx = None
    for i, det in enumerate(dets):
        if det[1] == det_id:
            debug_det = det
            det_idx = i
            break
    if debug_det is None:
        return
    
    x1,y1,w,h = debug_det[2:6]
    x2,y2 = x1 + w, y1 + h
    draw_detection(frame, x1,y1,x2,y2,debug_det[-1],debug_det[-2],color_map={debug_det[-1]: (255,255,0)})

    for i, track in enumerate(iou_tracks):
        if track.track_id == track_id:
            print(f"frame no {frame_no} track_id: {track_id} det_id: {det_id} det conf {debug_det[-2]} iou: {ious[i][det_idx]}")
            break

def print_track_debug_info(track_id, curr_tracks, curr_lost_tracks, curr_dets, prev_tracks, prev_lost_tracks):
    curr_track = None
    for track in curr_tracks:
        if track.track_id == track_id:
            curr_track = track
            break
    
    if not curr_track is None:
        print(f"Current track {track_id} associated with detection {curr_track.detection_id} with conf {curr_track.prob}")
        curr_associated_det = None
        for det in curr_dets:
            if det[1] == curr_track.detection_id:
                curr_associated_det = det
                break
        
        prev_track = None
        lost_prev_track = False
        for track in prev_tracks:
            if track.track_id == track_id:
                prev_track = track
                break
        if prev_track is None:
            for track in prev_lost_tracks:
                if track.track_id == track_id:
                    prev_track = track
                    lost_prev_track = True
                    break
        if lost_prev_track:
            print(f"Track {track_id} is lost in previous frame")
        
        if prev_track is None:
            print(f"Track {track_id} not found in previous tracks or previous lost tracks!")
        else:
            prev_track_box = [prev_track.x, prev_track.y, prev_track.x + prev_track.width, prev_track.y + prev_track.height]
            # curr_track_box = [curr_track.x, curr_track.y, curr_track.width, curr_track.height]
            curr_det_box = [curr_associated_det[2], curr_associated_det[3], curr_associated_det[2] + curr_associated_det[4], curr_associated_det[3] + curr_associated_det[5]]

            iou = torchvision.ops.box_iou(torch.tensor(prev_track_box).unsqueeze(0), torch.tensor(curr_det_box).unsqueeze(0))[0][0].item()
            diou, aspect_ratio_weight, giou = calc_ciou(prev_track_box[0], prev_track_box[1], prev_track_box[2], prev_track_box[3], curr_det_box[0], curr_det_box[1], curr_det_box[2], curr_det_box[3])
            alpha = aspect_ratio_weight / (1.0 - iou + aspect_ratio_weight) if iou > 0.0 else 0.0
            ciou = iou - diou - alpha * aspect_ratio_weight
            scaled_ciou = (max(ciou, -1.0) + 1.0) / 2.0
            print(f"Track {track_id} has iou {iou:4f} with matched box. diou {diou:4f} alpha {alpha:4f} v {aspect_ratio_weight:4f} aspect_ratio_weight {aspect_ratio_weight * alpha:4f} ciou {ciou:4f} scaled_ciou {scaled_ciou:4f} giou {giou:4f}")
    else:
        curr_track = None
        for track in curr_lost_tracks:
            if track.track_id == track_id:
                curr_track = track
                break
        if curr_track is None:
            print(f"Track {track_id} not found in current tracks or current lost tracks!")
            return
        print(f"Track {track_id} is lost in current frame, last known association {curr_track.detection_id} with conf {curr_track.prob}")

        formatted_dets = curr_dets[:, 2:6]
        formatted_dets[:, 2] = formatted_dets[:, 0] + formatted_dets[:, 2]
        formatted_dets[:, 3] = formatted_dets[:, 1] + formatted_dets[:, 3]
        formatted_dets = torch.tensor(formatted_dets)
        track_box = [curr_track.x, curr_track.y, curr_track.x + curr_track.width, curr_track.y + curr_track.height]

        ious = torchvision.ops.box_iou(torch.tensor(track_box).unsqueeze(0), formatted_dets)
        max_iou_idx = torch.argmax(ious[0])
        iou = ious[0][max_iou_idx]
        diou, aspect_ratio_weight, giou = calc_ciou(track_box[0], track_box[1], track_box[2], track_box[3], formatted_dets[max_iou_idx][0], formatted_dets[max_iou_idx][1], formatted_dets[max_iou_idx][2], formatted_dets[max_iou_idx][3])
        alpha = aspect_ratio_weight / (1.0 - iou + aspect_ratio_weight) if iou > 0.0 else 0.0
        ciou = iou - diou - alpha * aspect_ratio_weight
        scaled_ciou = (max(ciou, -1.0) + 1.0) / 2.0

        print(f"Track {track_id} has max iou {iou:4f} with box {curr_dets[max_iou_idx][1]} and conf {curr_dets[max_iou_idx][6]:4f}. diou {diou:4f} alpha {alpha:4f} v {aspect_ratio_weight:4f} aspect_ratio_weight {aspect_ratio_weight * alpha:4f} ciou {ciou:4f} scaled_ciou {scaled_ciou:4f} giou {giou:4f}")
   
def letterbox(img, new_shape=(640,640), color=(114, 114, 114), pad_amount=0):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left+pad_amount, right+pad_amount, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)

def get_dets_from_json(vid_frames_to_det_frames, det_by_frame, width=1920, height=1088, class_name_to_num={"Person": 0, "Bag": 1}):
    next_det_id = 0
    all_dets = []
    for frame_no in range(len(vid_frames_to_det_frames)):
        if vid_frames_to_det_frames[frame_no] is None:
            all_dets.append([])
            continue
        else:
            dets = []
            for det in det_by_frame[vid_frames_to_det_frames[frame_no]]:
                roi = det['roi']
                x1, y1 = roi['left']*width, roi['top']*height
                x2, y2 = roi['right']*width, roi['bottom']*height
                cls = class_name_to_num[det["class"]]
                conf = det["confidence"]
                # det_id = det["id"]
                det_id = next_det_id
                next_det_id += 1
                dets.append([frame_no, det_id, x1,y1,x2-x1,y2-y1, conf, cls])
            all_dets.append(dets)
    return all_dets

def get_dets_from_torch(det_path):
    cached_dets = torch.load(det_path).numpy()

    all_dets = []
    next_det_id = 0
    for i in range(len(cached_dets)):
        dets = []
        for det in cached_dets[i]:
            w,h = det[2] - det[0], det[3] - det[1] 
            dets.append([i, next_det_id, det[0], det[1], w, h, det[4], det[5]])
            next_det_id += 1
        all_dets.append(dets)
    print(f"Found {len(all_dets)} frames of detections, assuming this length for video")
    return all_dets

def main(video_path, det_path, out_video_name, start_frame=0, end_frame=np.inf, sensor_id=120, use_regen_pcp=False):
    assert os.path.exists(video_path)
    assert os.path.exists(det_path)

    if not use_regen_pcp:
        det_by_frame, det_frame_timestamps, vid_timestamps, missed_frames = parse_sensor_detections(det_path, video_path, sensor_id=sensor_id)
        vid_frames_to_det_frames = align_detection_to_video(det_frame_timestamps, vid_timestamps, thresh=.01)
        all_dets = get_dets_from_json(vid_frames_to_det_frames, det_by_frame)
    else:
        all_dets = get_dets_from_torch(det_path)
        vid_timestamps = np.ones(len(all_dets))

    bytetrack_human = ByteTrackerPy(
        # frame_rate=10,
        # track_buffer=5,
        # track_thresh=0.5, 
        # high_thresh=0.7,
        # match_thresh=0.7,

        # max_time_lost = num of update calls before kill track = frame_rate * track_buffer / 30.0
        frame_rate=30,
        track_buffer=30,
        
        # Low conf tracks are < track_thresh, high conf tracks are >= track_thresh
        track_thresh=0.5, 
        # Spawn track conf thresh
        high_thresh=0.7,
        
        # Matching costs
        # Costs are iou * iou_weight + score_normalized * (1-iou_weight)
        use_ciou=False, 
        high_conf_match_iou_weight=1.0,
        high_conf_match_min_iou=0.3,
        low_conf_match_iou_weight=1.0,
        low_conf_match_min_iou=0.5,
        track_activation_iou_weight=1.0,
        track_activation_min_iou=0.3,

        # Kalman args
        kalman_std_weight_pos = 1. / 20.,
        kalman_std_weight_vel = 1. / 160.,
        kalman_std_weight_position_meas = 1. / 20.,
        kalman_std_weight_position_mot = 1. / 20.,
        kalman_std_weight_velocity_mot = 1. / 160.,
        kalman_std_aspect_ratio_init = 3e-2,
        kalman_std_d_aspect_ratio_init = 3e-5,
        kalman_std_aspect_ratio_mot = 1e-2,
        kalman_std_d_aspect_ratio_mot = 1e-5,
        kalman_std_aspect_ratio_meas = 5e-2,
    )
    bytetrack_bag = ByteTrackerPy(
        # frame_rate=10,
        # track_buffer=5,
        # track_thresh=0.5, 
        # high_thresh=0.7,
        # match_thresh=0.7,

        # max_time_lost = num of update calls before kill track = frame_rate * track_buffer / 30.0
        frame_rate=30,
        track_buffer=30,
        
        # Low conf tracks are < track_thresh, high conf tracks are >= track_thresh
        track_thresh=0.3,
        # Spawn track conf thresh
        high_thresh=0.45,
        # Matching costs
        # Costs are iou * iou_weight + score_normalized * (1-iou_weight)
        use_ciou=True, 
        high_conf_match_iou_weight=1.0,
        high_conf_match_min_iou=0.5,
        low_conf_match_iou_weight=1.0,
        low_conf_match_min_iou=0.65,
        track_activation_iou_weight=1.0,
        track_activation_min_iou=0.5,

        # Kalman args
        kalman_std_weight_pos = 1. / 20.,
        kalman_std_weight_vel = 1. / 160.,
        kalman_std_weight_position_meas = 1. / 30.,
        kalman_std_weight_position_mot = 1. / 15.,
        kalman_std_weight_velocity_mot = 1. / 80.,
        kalman_std_aspect_ratio_init = 1e-1,
        kalman_std_d_aspect_ratio_init = 9e-5,
        kalman_std_aspect_ratio_mot = 1e-2,
        kalman_std_d_aspect_ratio_mot = 1e-5,
        kalman_std_aspect_ratio_meas = 9e-2,
    )
    # Open the raw H.265 video file
    cap = cv2.VideoCapture(video_path)
    # Check if the file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")

    vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = 1920
    # height = 1920
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_name, fourcc, vid_fps, (int(width*1.5),int(height*1.5)))
    bag_only_out = cv2.VideoWriter(out_video_name.replace('.mp4', '_bag_only.mp4'), fourcc, vid_fps, (int(width*1.5),int(height*1.5)))

    class_name_to_num = {
        "Person": 0,
        "Bag": 1,
    }
    class_idx_to_color = {
        0: (0,255,0),
        1: (255,0,255),
        "Person": (0,255,0),
        "Bag": (255,0,255),
    }

    threshold_points = np.array([
        [685, 1016], # BL
        [1232, 1079],# BR
        [1257, 309], # TR
        [1082, 302], # TL 
    ], dtype=np.int32)
    # threshold_points[:, 0] /= float(width)
    # threshold_points[:, 1] /= float(height)
    # Down is allowed direction of flow
    quad_alert_geometry_down = QuadAlertGeometry(
        threshold_points=threshold_points,
        radius=30,
    )
    # Up is allowed direction of flow
    quad_alert_geometry_up = QuadAlertGeometry(
        threshold_points=threshold_points,
        edge_idx=[2,3],
        radius=30,
    )

    frame_no = 0
    result_tracks = []
    result_statuses_up = []
    result_statuses_down = []
    result_lost_tracks = []
    prev_cand_bags = []
    prev_cand_humans = []
    prev_bag_tracks = []
    prev_person_tracks = []
    prev_lost_bag_tracks = []
    prev_lost_human_tracks = []
    bag_track_statuses_up = {}
    bag_track_statuses_down = {}
    human_track_statuses_up = {}
    human_track_statuses_down = {}
    error_counter = 0
    while cap.isOpened() and frame_no < len(vid_timestamps):
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"No frame was read, trying again")
            error_counter += 1
            if error_counter > 10:
                print(f"Error counter > 10, breaking")
                break
            continue
        if frame_no + 1 == len(vid_timestamps):
            print(f"Reached end of video, breaking")
            frame_no += 1
            break
        error_counter = 0
        # Convert BGR (OpenCV default) to RGB for matplotlib
        # letter_box_frame, _, _ = letterbox(frame_bgr, new_shape=(1920,1920))
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cv2.putText(
            frame,
            f"Frame: {frame_no}",
            (10, 30),  # Position at top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),  # Green color
            2  # Thickness
        )

        raw_det_frame = frame.copy()
        post_process_det_frame = frame.copy()
        lost_track_frame = frame.copy()
        frame_dets = all_dets[frame_no]
            
        # Draw raw detections
        raw_det_frame_bag = raw_det_frame.copy()
        for det in frame_dets:
            x1,y1,w,h = det[2:6]
            cls, conf = det[-1], det[-2]
            if conf > .1:
                if cls == 1:
                    draw_detection(raw_det_frame_bag, x1,y1,x1+w,y1+h,cls,conf,color_map=class_idx_to_color, iou=int(det[1]))
                draw_detection(raw_det_frame, x1,y1,x1+w,y1+h,cls,conf,color_map=class_idx_to_color, iou=int(det[1]))
        
        # Postprocess detections
        postprocessed_dets = postprocess_boxes(frame_dets, min_conf_thresh_bag=.1, min_conf_thresh_human=.1, nms_iou_bag=.65, nms_iou_human_bag=.85)
        post_process_det_frame_bag = post_process_det_frame.copy()
        for det in postprocessed_dets:
            x1,y1,w,h = det[2:6]
            cls, conf = det[-1], det[-2]
            if cls == 1:
                draw_detection(post_process_det_frame_bag, x1,y1,x1+w,y1+h,cls,conf,color_map=class_idx_to_color,iou=int(det[1]))    
            draw_detection(post_process_det_frame, x1,y1,x1+w,y1+h,cls,conf,color_map=class_idx_to_color,iou=int(det[1]))

        if len(postprocessed_dets):
            human_dets = postprocessed_dets[postprocessed_dets[:, -1] == 0]
            bag_dets = postprocessed_dets[postprocessed_dets[:, -1] == 1]
        else:
            human_dets = []
            bag_dets = []

        # Tracker update
        person_tracks = bytetrack_human.update(human_dets)
        bag_tracks = bytetrack_bag.update(bag_dets)

        # Track status update
        # TODO are track ids unique across classes?
        quad_alert_geometry_down.draw_threshold(frame, color=(255,0,255), thickness=2, draw_grid_lines=False)
        # quad_alert_geometry_up.draw_threshold(frame, color=(255,0,255), thickness=2, draw_grid_lines=False)
        frame_track_statuses_down = {}
        frame_track_statuses_up = {}
        for track in bag_tracks:
            previous_track_status_down = bag_track_statuses_down.get(track.track_id, 0)
            previous_track_status_up = bag_track_statuses_up.get(track.track_id, 0)
            # Bottom center of the track
            pos = track.x + track.width / 2, track.y + track.height
            alert_status_down, inner_prod_down, in_quad = quad_alert_geometry_down.get_track_status(pos, track.track_vel_xy, previous_track_status_down, frame, debug_plotting=True)
            alert_status_up, inner_prod_up, in_quad = quad_alert_geometry_up.get_track_status(pos, track.track_vel_xy, previous_track_status_up, frame, debug_plotting=False)

            bag_track_statuses_down[track.track_id] = alert_status_down
            bag_track_statuses_up[track.track_id] = alert_status_up
            frame_track_statuses_down[f"{track.track_id},1"] = [alert_status_down, inner_prod_down, int(in_quad)]
            frame_track_statuses_up[f"{track.track_id},1"] = [alert_status_up, inner_prod_up, int(in_quad)]
        for track in person_tracks:
            previous_track_status_down = human_track_statuses_down.get(track.track_id, 0)
            previous_track_status_up = human_track_statuses_up.get(track.track_id, 0)
            # Bottom center of the track
            pos = track.x + track.width / 2, track.y + track.height
            alert_status_down, inner_prod_down, in_quad = quad_alert_geometry_down.get_track_status(pos, track.track_vel_xy, previous_track_status_down, frame, debug_plotting=True)
            alert_status_up, inner_prod_up, in_quad = quad_alert_geometry_up.get_track_status(pos, track.track_vel_xy, previous_track_status_up, frame, debug_plotting=False)

            human_track_statuses_down[track.track_id] = alert_status_down
            human_track_statuses_up[track.track_id] = alert_status_up
            frame_track_statuses_down[f"{track.track_id},0"] = [alert_status_down, inner_prod_down, int(in_quad)]
            frame_track_statuses_up[f"{track.track_id},0"] = [alert_status_up, inner_prod_up, int(in_quad)]
        result_statuses_down.append(frame_track_statuses_down.copy())
        result_statuses_up.append(frame_track_statuses_up.copy())

        for track in bag_tracks:
            shade = bag_track_statuses_down.get(track.track_id, 0)
            draw_track(frame, track, shade, color=class_idx_to_color[1], draw_corners=False)
        bag_track_frame = frame.copy()
        for track in person_tracks:
            shade = human_track_statuses_down.get(track.track_id, 0)
            draw_track(frame, track, shade, color=class_idx_to_color[0], draw_corners=False)
        
        extracted_person_tracks = [extract_track_info(track, 0) for track in person_tracks]
        extracted_bag_tracks = [extract_track_info(track, 1) for track in bag_tracks]
        result_tracks.append(extracted_person_tracks + extracted_bag_tracks)

        # Draw lost tracks for debugging
        lost_bags = bytetrack_bag.tracker.get_lost_tracks()
        lost_humans = bytetrack_human.tracker.get_lost_tracks()
        cand_bags = bytetrack_bag.tracker.get_inactive_tracks()
        cand_humans = bytetrack_human.tracker.get_inactive_tracks()

        lost_bag_ious = calc_ious(lost_bags, bag_dets, calc_ciou=True)
        lost_human_ious = calc_ious(lost_humans, human_dets, calc_ciou=False)
        cand_bag_ious = calc_ious(prev_cand_bags, bag_dets, calc_ciou=True)
        cand_human_ious = calc_ious(prev_cand_humans, human_dets, calc_ciou=False)

        all_bag_ious = calc_ious(prev_bag_tracks + prev_lost_bag_tracks, bag_dets, calc_ciou=True)
        pure_iou = calc_ious(prev_bag_tracks + prev_lost_bag_tracks, bag_dets, calc_ciou=False)

        # draw_debug_detection(frame, frame_no, 69, bag_tracks, bag_dets, all_bag_ious, prev_bag_tracks + prev_lost_bag_tracks)
        # draw_debug_detection(frame, frame_no, 69, bag_tracks, bag_dets, pure_iou, prev_bag_tracks + prev_lost_bag_tracks)

        if len(lost_bag_ious):
            for i, track in enumerate(lost_bags):
                iou = lost_bag_ious[i].max().item()
                draw_track(lost_track_frame, track, 0, color=class_idx_to_color[1], draw_corners=False, iou=f" I {iou:.2f}", draw_prob=False)
        if len(cand_bag_ious):
            for i, track in enumerate(prev_cand_bags):
                iou = cand_bag_ious[i].max().item()
                draw_track(lost_track_frame, track, 1, color=class_idx_to_color[1], draw_corners=False, iou=f" I {iou:.2f}", draw_prob=False)
        lost_track_frame_bag = lost_track_frame.copy()

        if len(lost_human_ious):
            for i, track in enumerate(lost_humans):
                iou = lost_human_ious[i].max().item()
                draw_track(lost_track_frame, track, 0, color=class_idx_to_color[0], draw_corners=False, iou=f" I {iou:.2f}", draw_prob=False)
        if len(cand_human_ious):
            for i, track in enumerate(prev_cand_humans):
                iou = cand_human_ious[i].max().item()
                draw_track(lost_track_frame, track, 1, color=class_idx_to_color[0], draw_corners=False, iou=f" I {iou:.2f}", draw_prob=False)
        
        extracted_lost_humans = [extract_track_info(track, 0) for track in lost_humans]
        extracted_lost_bags = [extract_track_info(track, 1) for track in lost_bags]
        result_lost_tracks.append(extracted_lost_humans + extracted_lost_bags)

        prev_cand_bags = cand_bags
        prev_cand_humans = cand_humans
        prev_bag_tracks = bag_tracks
        prev_person_tracks = person_tracks
        prev_lost_bag_tracks = lost_bags
        prev_lost_human_tracks = lost_humans

        if frame_no % 500 == 0:
            print(frame_no)

        # Skip frames for viz (still run all the other tracker stuff)
        if frame_no < start_frame:
            frame_no += 1
            continue
        if frame_no >= end_frame:
            frame_no += 1
            break
        
        display_frame_top = cv2.hconcat([frame, post_process_det_frame])
        display_frame_bot = cv2.hconcat([lost_track_frame, raw_det_frame])
        display_frame = cv2.vconcat([display_frame_top, display_frame_bot])
        out.write(cv2.resize(display_frame[:,:,::-1], (int(width*1.5),int(height*1.5))))

        display_frame_bag_top = cv2.hconcat([bag_track_frame, post_process_det_frame_bag])
        display_frame_bag_bot = cv2.hconcat([lost_track_frame_bag, raw_det_frame_bag])
        display_frame_bag = cv2.vconcat([display_frame_bag_top, display_frame_bag_bot])
        bag_only_out.write(cv2.resize(display_frame_bag[:,:,::-1], (int(width*1.5),int(height*1.5))))
        frame_no += 1

    cap.release()
    out.release()
    print(f"Wrote video to {out_video_name}")
    print(frame_no, len(vid_timestamps))
    return result_tracks, result_statuses_down, result_statuses_up, result_lost_tracks, frame_no

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run resim tracker on video')
    parser.add_argument('--det_path', required=True, help='Path to detections file')
    parser.add_argument('--video_path', required=True, help='Path to video file')
    parser.add_argument('--out_video', required=True, help='Output video path')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame number')
    parser.add_argument('--end_frame', type=str, default="np.inf", help='End frame number')
    parser.add_argument('--sensor_id', type=int, required=True, help='Sensor ID')
    args = parser.parse_args()

    end_frame = eval(args.end_frame)

    assert os.path.exists(args.video_path)
    assert os.path.exists(args.det_path)

    result_tracks, statuses_down, statuses_up, lost_tracks, frame_no = main(
        args.video_path, 
        args.det_path, 
        args.out_video, 
        start_frame=args.start_frame, 
        end_frame=end_frame, 
        sensor_id=args.sensor_id, 
        use_regen_pcp=args.det_path.endswith('.pt')
    )
    

    with open(args.out_video.replace('.mp4', '.json'), 'w') as f:
        json.dump(result_tracks, f)
        print(f"Wrote results to {args.out_video.replace('.mp4', '.json')}")

    with open(args.out_video.replace('tracks.mp4', 'lost_tracks.json'), 'w') as f:
        json.dump(lost_tracks, f)
        print(f"Wrote lost tracks to {args.out_video.replace('tracks.mp4', 'lost_tracks.json')}")

    with open(args.out_video.replace('.mp4', '_track_statuses_down.json'), 'w') as f:
        json.dump(statuses_down, f)
        print(f"Wrote bag track statuses to {args.out_video.replace('.mp4', '_bag_track_statuses_down.json')}")

    with open(args.out_video.replace('.mp4', '_track_statuses_up.json'), 'w') as f:
        json.dump(statuses_up, f)
        print(f"Wrote bag track statuses to {args.out_video.replace('.mp4', '_bag_track_statuses_up.json')}")