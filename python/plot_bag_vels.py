import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import pandas as pd
import cv2
from drawing_tools import draw_track, draw_detection
import os
import matplotlib.animation as animation
from utils import VIDEOS_TO_NUM_FRAMES
from PIL import Image

def compute_displacement(x_positions, y_positions, areas, window_size=5):
    """
    Compute displacement between first and last frame in a sliding window.
    
    Parameters:
    - x_positions: list or np.array of x coordinates
    - y_positions: list or np.array of y coordinates
    - window_size: number of frames to compute displacement over
    
    Returns:
    - displacements: np.array of displacement values (same length as input)
    """
    x_positions = np.asarray(x_positions)
    y_positions = np.asarray(y_positions)
    
    if len(x_positions) != len(y_positions):
        raise ValueError("x_positions and y_positions must have same length")
    
    n = len(x_positions)
    displacements = np.zeros(n)
    normalized_displacements = np.zeros(n)
    
    # For positions before window_size, use available points
    for i in range(min(window_size, n)):
        dx = x_positions[i] - x_positions[0]
        dy = y_positions[i] - y_positions[0]
        displacements[i] = np.sqrt(dx*dx + dy*dy)
        avg_area = np.mean(areas[0:i+1])
        normalized_displacements[i] = displacements[i] / np.sqrt(avg_area)
    
    # For remaining positions, use full window
    for i in range(window_size, n):
        start_idx = i - window_size + 1
        dx = x_positions[i] - x_positions[start_idx]
        dy = y_positions[i] - y_positions[start_idx]
        displacements[i] = np.sqrt(dx*dx + dy*dy)
        avg_area = np.mean(areas[start_idx:i+1])
        normalized_displacements[i] = displacements[i] / np.sqrt(avg_area)
    
    return displacements, normalized_displacements

def find_consecutive_trues(bool_array, n_consecutive=5):
    """
    Find if there are n consecutive True values in a boolean array using convolution.
    
    Args:
        bool_array: numpy array of booleans
        n_consecutive: number of consecutive True values to look for
        
    Returns:
        numpy array of booleans, True where n consecutive True values start
    """
    assert n_consecutive % 2 == 1
    # Convert boolean array to int array (True -> 1, False -> 0)
    int_array = bool_array.astype(int)
    # Create kernel of n ones
    kernel = np.ones(n_consecutive)
    # Convolve and check where the result equals n
    convolved = np.convolve(int_array, kernel, mode='same')
    # Append n_consecutive // 2 zeros to start and throw away n_consecutive // 2 elements from end
    result = np.concatenate([np.zeros(n_consecutive // 2), convolved])
    result = result[:-(n_consecutive // 2)]
    return result == n_consecutive

def compute_is_thrown(track_vy_norm, track_displacements_norm, num_ticks, min_displacement, min_v):
    is_consistently_wrong_way = find_consecutive_trues(np.abs(track_vy_norm) > min_v, num_ticks)
    is_thrown = is_consistently_wrong_way & (track_displacements_norm > min_displacement)
    return is_thrown

def compute_is_alerting(track_vy_norm, track_d_ar, track_displacements, num_ticks_abs, num_ticks_v, min_v, max_d_ar, min_displacement):
    is_below_abs_thresh = track_d_ar < max_d_ar
    is_consistent_abs = find_consecutive_trues(is_below_abs_thresh, num_ticks_abs)
    # 0 out velocities that violate the aspect ratio so they are not considered to be alerting ticks
    track_vy_norm = np.abs(track_vy_norm)
    is_wrong_way = track_vy_norm > min_v
    is_consistently_wrong_way = find_consecutive_trues(is_wrong_way, num_ticks_v)
    is_alerting = is_consistently_wrong_way & is_consistent_abs & (track_displacements > min_displacement)
    return is_alerting, is_wrong_way, is_consistently_wrong_way, is_consistent_abs

def convert_track_results_to_df(track_results):
    all_tracks = {
        "frame": [],
        "aspect_ratio": [],
        "area": [],
        "x1": [],
        "y1": [],
        "x2": [],
        "y2": [],
        "vx": [],
        "vy": [],
        "cls": [],
        "track_id": [],
        "detection_id": [],
        "prob": [],
    }
    for frame_idx, frame_tracks in enumerate(track_results):
        for track in frame_tracks:
            all_tracks["frame"].append(frame_idx)
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            all_tracks["x1"].append(x1)
            all_tracks["y1"].append(y1)
            all_tracks["x2"].append(x2)
            all_tracks["y2"].append(y2)
            all_tracks["aspect_ratio"].append((x2 - x1) / (y2 - y1))
            all_tracks["area"].append((x2 - x1) * (y2 - y1))
            all_tracks["cls"].append(track[6])
            all_tracks["vx"].append(track[4])
            all_tracks["vy"].append(track[5])
            all_tracks["track_id"].append(track[7])
            all_tracks["detection_id"].append(track[8])
            all_tracks["prob"].append(track[9])
    df = pd.DataFrame(all_tracks)
    return df

def convert_alert_results_to_df(alert_results):
    all_alerts = {
        "frame": [],
        "track_id": [],
        "cls": [],
        "alert_status": [],
        "inner_prod": [],
        "in_quad": [],
    }
    for frame_idx, frame_alerts in enumerate(alert_results):
        for track_info, alert_info in frame_alerts.items():
            track_id, cls = track_info.split(",")
            alert_status, inner_prod, in_quad = alert_info
            all_alerts["frame"].append(frame_idx)
            all_alerts["track_id"].append(int(track_id))
            all_alerts["cls"].append(int(cls))
            all_alerts["alert_status"].append(alert_status)
            all_alerts["inner_prod"].append(inner_prod)
            all_alerts["in_quad"].append(in_quad)
    df = pd.DataFrame(all_alerts)
    return df

def load_all_video_frames_2(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_all_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = -1
    for vid_name, num_frames in VIDEOS_TO_NUM_FRAMES.items():
        if vid_name in video_path:
            n_frames = num_frames
            break
    if n_frames == -1:
        raise ValueError("Could not find video in VIDEOS_TO_NUM_FRAMES")
    all_frames = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        all_frames.append(frame)
        frame_idx += 1
    return all_frames

def create_track_visualization(track_df, video_path, output_path, fps=30):
    """Create a visualization combining track boxes from video and velocity plots
    
    Args:
        track_df (pd.DataFrame): DataFrame containing track history
        video_path (str): Path to original video file
        output_path (str): Path to save output visualization
        fps (int): Frames per second for output video
    """
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10
    
    # Initialize velocity plot
    frames = track_df['frame'].values
    vx_vals = track_df['vx'].values
    vy_vals = track_df['vy'].values
    
    # Setup matplotlib figure
    fig, (ax_video, ax_vel) = plt.subplots(1, 2, figsize=(12, 5))
    
    im = ax_video.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    ax_video.axis('off')
    ax_video.set_title("Video")
    ax_video.axis("off")
    
    line_vx, = ax_vel.plot(frames, vx_vals, 'b-', label='vx')
    line_vy, = ax_vel.plot(frames, vy_vals, 'r-', label='vy')
    # Create point objects for current velocity indicators
    vx_point, = ax_vel.plot([], [], 'bo', markersize=10)
    vy_point, = ax_vel.plot([], [], 'ro', markersize=10)
    ax_vel.set_xlabel('Frame')
    ax_vel.set_ylabel('Velocity')
    ax_vel.set_title('Track Velocities')
    ax_vel.grid(True)
    ax_vel.legend()

    def update(frame_idx):
        # Get current frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return [im, line_vx,line_vy, vx_point, vy_point]
            
        # Convert BGR to RGB for matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get track data for current frame
        curr_track = track_df[track_df['frame'] == frame_idx].iloc[0]
        x1, y1, x2, y2 = curr_track['x1'], curr_track['y1'], curr_track['x2'], curr_track['y2']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        # Display frame
        temp_frame = np.zeros_like(frame)
        temp_frame[..., 0] = 255
        im.set_array(temp_frame)
        
        # Update velocity points
        vx_point.set_data([frame_idx], [curr_track['vx']])
        vy_point.set_data([frame_idx], [curr_track['vy']])
        
        return [im, line_vx,line_vy, vx_point, vy_point]
        
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=frames,
        interval=1000 / fps,
        blit=True
    )
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Your Name'), bitrate=1800)
    
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer=writer)
    print(f"Finished saving animation to {output_path}")
    cap.release()
    plt.close()

def windowed_moving_average(data, window_size=5):
    """
    Apply a windowed moving average filter to smooth the data.

    Parameters:
    - data: list or np.array of input values
    - window_size: number of points to average over

    Returns:
    - filtered: np.array of filtered values (same length as input)
    """
    assert window_size % 2 == 1
    
    data = np.asarray(data)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    # Create convolution kernel
    kernel = np.ones(window_size) / window_size

    # Apply convolution (same mode keeps output the same length)
    smoothed = np.convolve(data, kernel, mode='valid')

    # Compute the missing elements
    missing_elements = []
    for i in range(min(window_size-1, len(data))):
        missing_elements.append(np.mean(data[0:i+1]))

    if len(data) < window_size:
        return np.array(missing_elements)

    results = np.concatenate([np.array(missing_elements), smoothed])
    return results

def ema(data, alpha=0.2):
    """
    Exponential moving average

    Parameters:
    - data: list or np.array of input values (e.g., velocity, area, etc.)
    - alpha: smoothing factor in [0, 1], controls the weight of the new data

    Returns:
    - filtered: list of filtered values
    """
    if len(data) == 0:
        return []

    filtered = [data[0]]  # initialize with the first value

    for t in range(1, len(data)):
        y_prev = filtered[-1]
        x_t = data[t]
        y_t = alpha * x_t + (1 - alpha) * y_prev
        filtered.append(y_t)

    return np.array(filtered)

def get_alert_frames(track_alerts, cls, id):
    alerts, frames = track_alerts[(cls, id)]
    print(frames[alerts.nonzero()])

def viz_alerting(track_df, track_id, cls, cls_translator, thresholds):
    frames = track_df["frame"].values
    # Compute diffs
    track_df["d_aspect_ratio"] = track_df["aspect_ratio"].diff().abs()
    track_df["d_area"] = track_df["area"].diff().abs()
    print(f"Visualizing track {track_id} cls {cls}")

    # Create figure with three subplots
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 12), height_ratios=[2, 1, 1, 1, 1, 1])
    
    # Plot velocities and inner products in top subplot
    ax1_twin = ax1.twinx()
    displacements, normalized_displacements = compute_displacement(track_df["x1"].values, track_df["y1"].values, track_df["area"].values, window_size=thresholds["num_ticks_v"])
    displ_long_wind, norm_displ_long_wind = compute_displacement(track_df["x1"].values, track_df["y1"].values, track_df["area"].values, window_size=21)
    ax1.plot(frames, normalized_displacements, 'b-', label='normalized displacement')
    ax1.plot(frames, norm_displ_long_wind, 'b-', label=f'norm displ w{21}', alpha=.25)
    ax1.plot(frames, np.ones_like(frames) * thresholds["min_displacement"], 'b-', label='min displacement')
    ax1.set_ylabel("Normalized Displacement")
    # ax1_twin.plot(frames, track_df["vx_norm"].values, 'g-', label='normalized sqrt vx')
    ax1_twin.plot(frames, track_df["vy_norm"].values, 'm-', label='normalized vy')
    # ax1_twin.plot(frames, windowed_moving_average(track_df["vy_norm"].values, 21), 'r-', label='moving avg w21', alpha=.5)
    
    # ax1_twin.plot(frames, ema(track_df["vy_norm"].values, 0.2), 'y-', label='ema .2')
    ax1_twin.plot(frames, np.ones_like(frames) * thresholds["min_v"], 'm-', label='min v')
    ax1_twin.plot(frames, np.ones_like(frames) * -thresholds["min_v"], 'm-')
    # ax1.plot(frames, track_df["inner_prod_up"].values, 'g-', label='inner_prod_up', alpha=0.5)
    # ax1.plot(frames, track_df["inner_prod_down"].values, 'magenta', label='inner_prod_down', alpha=0.5)
    ax1_twin.set_ylabel("Normalized Velocity")
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot aspect ratio in middle subplot with two y-axes
    aspect_ratios = track_df["aspect_ratio"].values
    delta_aspect_ratios = track_df["d_aspect_ratio"].values
    # Primary y-axis for aspect ratio
    ax2.plot(frames, aspect_ratios, 'm', label='aspect_ratio')
    ax2.set_ylabel("Aspect Ratio")
    ax2.grid(True)
    # Secondary y-axis for delta aspect ratio
    ax2_twin = ax2.twinx()
    ax2_twin.plot(frames, np.abs(delta_aspect_ratios), 'g-', label='abs_delta_aspect_ratio')
    ax2_twin.plot(frames, np.ones_like(frames) * thresholds["max_d_ar"], 'g-')
    ax2_twin.set_ylim(0, .075)
    ax2_twin.set_ylabel("Abs Delta Aspect Ratio")
    # Add legends for both axes in middle subplot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Plot box area in bottom subplot with two y-axes
    areas = track_df["area"].values
    delta_areas = track_df["d_area"].values
    # Primary y-axis for area
    ax3.plot(frames, areas, 'blue', label='area')
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Box Area (pixels²)")
    ax3.grid(True)
    # Secondary y-axis for delta area
    ax3_twin = ax3.twinx()
    ax3_twin.plot(frames, np.abs(delta_areas), 'red', label='abs_delta_area')
    ax3_twin.set_ylabel("Abs Delta Area")
    # Add legends for both axes in bottom subplot
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

    is_alerting, is_wrong_way, is_consistently_wrong_way, is_consistent_abs = compute_is_alerting(
        track_df["vy_norm"].values, 
        track_df["d_aspect_ratio"].values, 
        displacements, 
        thresholds["num_ticks_abs"], 
        thresholds["num_ticks_v"], 
        thresholds["min_v"], 
        thresholds["max_d_ar"], 
        thresholds["min_displacement"]
    )
    if cls == 1:
        is_thrown = compute_is_thrown(
            track_df["vy_norm"].values,
            norm_displ_long_wind,
            thresholds["num_ticks_v_fast_obj"],
            thresholds["min_displacement_fast_obj"],
            thresholds["min_v_fast_obj"]
        )
        ax4.plot(frames, is_thrown, 'b-', label='is_thrown')
    else:
        is_thrown = np.zeros_like(frames)

    ax4.plot(frames, is_alerting, 'r-', label='is_alerting')
    ax4.plot(frames, is_wrong_way, 'g-', label='is_wrong_way', alpha=.25)
    # ax4.plot(frames, is_consistently_wrong_way, 'b-', label='is_consistently_wrong_way', alpha=.25)
    ax4.plot(frames, ~is_consistent_abs, 'm-', label='invalid_d_abs', alpha=.25)
    ax4.legend(loc='upper right')

    # Viz normalization stuff
    vx_vals = track_df["vx"].values
    vy_vals = track_df["vy"].values
    ax5_twin = ax5.twinx()
    ax5.plot(frames, vx_vals, 'b-', label='vx', alpha=.25)
    ax5.plot(frames, vy_vals, 'r-', label='vy', alpha=.25)
    ax5_twin.plot(frames, track_df["vx_norm"].values, 'b-', label='vx_norm')
    ax5_twin.plot(frames, track_df["vy_norm"].values, 'r-', label='vy_norm')
    ax5.set_ylabel("Velocity")
    ax5_twin.set_ylabel("Normalized Velocity")
    ax5.grid(True)
    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines5 + lines6, labels5 + labels6, loc='upper right')

    ax6_twin = ax6.twinx()
    ax6.plot(frames, displacements, 'r-', label='displacement')
    ax6_twin.plot(frames, normalized_displacements, 'g-', label='normalized displacement')
    ax6.set_ylabel("Displacement")
    ax6_twin.set_ylabel("Normalized Displacement")
    ax6.grid(True)
    lines7, labels7 = ax6.get_legend_handles_labels()
    lines8, labels8 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines7 + lines8, labels7 + labels8, loc='upper right')
    
    # Set title for entire figure
    fig.suptitle(f"Track {track_id} cls {cls_translator[cls]}")
    # Adjust layout and save
    plt.tight_layout()
    # plt.savefig(output_path)
    # plt.close()

    return is_alerting, is_thrown

def main():
    parser = argparse.ArgumentParser(description='Plot velocity history for a specific track')
    parser.add_argument('--track_results', required=True, help='Path to tracker results JSON file')
    parser.add_argument('--video_path', required=True, help='Path to original video file')
    parser.add_argument('--output_dir', required=False, default="./track_analysis/", help='Output path for velocity plot')
    args = parser.parse_args()

    alert_results_path_up = args.track_results.replace(".json", "_track_statuses_up.json")
    alert_results_path_down = args.track_results.replace(".json", "_track_statuses_down.json")

    # Load tracking results
    with open(args.track_results, 'r') as f:
        track_results = json.load(f)
    with open(alert_results_path_up, 'r') as f:
        alert_results_up = json.load(f)
    with open(alert_results_path_down, 'r') as f:
        alert_results_down = json.load(f)
    
    import pdb; pdb.set_trace()

    alert_df_up = convert_alert_results_to_df(alert_results_up)
    alert_df_down = convert_alert_results_to_df(alert_results_down)
    full_df = convert_track_results_to_df(track_results)

    # Add in_quad and inner_prod to track_df
    alert_df_up = alert_df_up.rename(columns={"inner_prod": "inner_prod_up"})
    alert_df_down = alert_df_down.rename(columns={"inner_prod": "inner_prod_down"})
    full_df = full_df.merge(alert_df_up[["frame", "track_id", "cls", "inner_prod_up", "in_quad"]], on=["frame", "track_id", "cls"], how="left")
    full_df = full_df.merge(alert_df_down[["frame", "track_id", "cls", "inner_prod_down"]], on=["frame", "track_id", "cls"], how="left")

    video_frames = load_all_video_frames(args.video_path)

    class_translator = {
        0: "person",
        1: "bag",
    }
    tracks_to_viz = [
        # Tracks that are static at some point
        (0, 19), # moving then static, occluded at some point
        (0, 28), # moving then static
        (1, 35), # swinging but static
        (1, 37), # static backpack, lost / occluded at some point
        (1, 40), # occluded, static, swining, handed off
        (1, 41), # backpack strap, wild bboxes
        (1, 47), # suitcase slid then static
        (1, 51), # picked up then static
        (1, 56), # mostly static, keller moving around / squating etc
        # Normal movement tracks
        (0, 3),
        (0, 4),
        (0, 8),
        (0, 7),
        (1, 2),
        (1, 4),
        (1, 5),
        (1, 14),
        # Swung bags
        (1, 6),
        (1, 8),
        (1, 11),
        # Occlusion
        (0, 15),
        (0, 27),
        (1, 16),
        # Thrown bag
        (1, 55),
        (1, 57),
        (1, 61), # slid
        # Flickery box
        (1, 66), # stacked suitcase
    ]
    thresholds_bag = {
        # For more nominal case
        "min_v": .01,
        "max_d_ar": .02,
        "min_displacement": .2,
        "num_ticks_abs": 7,
        "num_ticks_v": 7,
        # For fast objects
        "num_ticks_v_fast_obj": 3,
        "min_displacement_fast_obj": .5,
        "min_v_fast_obj": .15,
    }
    # Roughly tuned for 1 step
    thresholds_person = {
        "min_v": .01,
        "max_d_ar": .02,
        "min_displacement": .09,
        "num_ticks_abs": 7,
        "num_ticks_v": 7,
    }

    full_df["vx_norm"] = full_df["vx"] / np.sqrt(full_df["area"])
    full_df["vy_norm"] = full_df["vy"] / np.sqrt(full_df["area"])
    d_areas = []
    d_aspect_ratios = []
    vys = []
    track_alerts = {}
    for cls, cls_df in full_df.groupby("cls"):
        thresholds = thresholds_bag if cls == 1 else thresholds_person
        for track_id, track_df in cls_df.groupby("track_id"):
            # # Fill gap frames
            # frames = track_df["frame"].values
            # full_index = range(frames[0], frames[-1] + 1)
            # track_df = track_df.reindex(full_index)
            track_df = track_df.reset_index(drop=True)

            # Compute diffs
            track_df["d_aspect_ratio"] = track_df["aspect_ratio"].diff().abs()
            track_df["d_area"] = track_df["area"].diff().abs()
            d_areas.extend(track_df["d_area"].values.tolist())
            d_aspect_ratios.extend(track_df["d_aspect_ratio"].values.tolist())
            vys.extend(track_df["vy_norm"].values.tolist())
            if (cls, track_id) not in tracks_to_viz:
                continue
            if len(track_df) < 2:
                print(f"Track {track_id} cls {cls} has less than 2 frames")
                continue

            is_alerting, is_thrown = viz_alerting(track_df, track_id, cls, class_translator, thresholds)
            frames = track_df["frame"].values

            output_path = os.path.join(args.output_dir, f"cls_{class_translator[cls]}_track_{track_id}.png")
            plt.savefig(output_path)
            plt.close()
            # create_track_visualization(track_df, args.video_path, output_path)
            track_alerts[(cls, track_id)] = is_alerting, frames

            gif_frames = []
            for frame_idx in range(frames[0], frames[-1] + 1):
                frame = video_frames[frame_idx]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.putText(
                    frame_rgb,
                    f"Frame: {frame_idx}",
                    (10, 30),  # Position at top-left
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # Font scale
                    (0, 255, 0),  # Green color
                    2  # Thickness
                )
                row = track_df[track_df["frame"] == frame_idx]
                if len(row) != 0:
                    idx = row.index.tolist()[0]
                    x1, y1, x2, y2 = row["x1"].values[0], row["y1"].values[0], row["x2"].values[0], row["y2"].values[0]
                    track_info = [x1, y1, x2, y2, row["vx"].values[0], row["vy"].values[0], row["track_id"].values[0]]
                    shade = 1 if is_alerting[idx] else 0
                    color = (0, 255, 0) if cls == 0 else (255, 0, 255)
                    draw_track(frame_rgb, track_info, shade, color)
                gif_frames.append(Image.fromarray(frame_rgb))

            gif_frames[0].save(os.path.join(args.output_dir, f"cls_{class_translator[cls]}_track_{track_id}.gif"), format="GIF", append_images=gif_frames[1:], save_all=True, duration=100, loop=0)
    
    plt.scatter(d_aspect_ratios, vys)
    plt.xlabel("Delta Aspect Ratio")
    plt.ylabel("Velocity")
    plt.title("Velocity vs Delta Aspect Ratio for all tracks")
    plt.savefig(os.path.join(args.output_dir, "velocity_vs_delta_aspect_ratio.png"))
    plt.close()

    plt.scatter(d_areas, vys)
    plt.xlabel("Delta Area")
    plt.ylabel("Velocity")
    plt.title("Velocity vs Delta Area for all tracks")
    plt.savefig(os.path.join(args.output_dir, "velocity_vs_delta_area.png"))
    plt.close()

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()