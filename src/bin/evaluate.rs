//! CLI evaluation script for comparing tracker results to ground truth.
//!
//! Usage:
//!   cargo run --bin evaluate --release -- \
//!     <detections.json> <timestamps.json> \
//!     [--tracker-fps 5.0] \
//!     [--track-thresh 0.25] \
//!     [--high-thresh 0.5] \
//!     [--track-buffer 1.0] \
//!     [--iou-threshold 0.5]

use std::env;
use std::path::Path;

use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::dataset::Clip;
use jamtrack_rs::evaluation::{Evaluator, GtDetection, PredDetection};
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

fn print_usage() {
    eprintln!("Usage: evaluate <detections.json> <timestamps.json> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --tracker-fps <fps>      Target FPS for tracker (0 = native) [default: 0]");
    eprintln!("  --track-thresh <thresh>  Min confidence for low-conf pool [default: 0.25]");
    eprintln!("  --high-thresh <thresh>   Min confidence to spawn new track [default: 0.5]");
    eprintln!("  --track-buffer <secs>    Track buffer in seconds [default: 1.0]");
    eprintln!("  --iou-threshold <iou>    IoU threshold for evaluation [default: 0.5]");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let detections_path = &args[1];
    let timestamps_path = &args[2];

    // Parse optional arguments
    let mut tracker_fps = 0.0f32;  // 0 = use native FPS
    let mut track_thresh = 0.25f32;
    let mut high_thresh = 0.5f32;
    let mut track_buffer_secs = 1.0f32;
    let mut iou_threshold = 0.5f32;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--tracker-fps" => {
                i += 1;
                tracker_fps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--track-thresh" => {
                i += 1;
                track_thresh = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.25);
            }
            "--high-thresh" => {
                i += 1;
                high_thresh = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            }
            "--track-buffer" => {
                i += 1;
                track_buffer_secs = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            "--iou-threshold" => {
                i += 1;
                iou_threshold = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if !Path::new(detections_path).exists() {
        eprintln!("Error: detections file not found: {}", detections_path);
        std::process::exit(1);
    }
    if !Path::new(timestamps_path).exists() {
        eprintln!("Error: timestamps file not found: {}", timestamps_path);
        std::process::exit(1);
    }

    // Load clip using the shared dataset module
    let clip = Clip::load_human_labeled(Path::new(detections_path), Path::new(timestamps_path))
        .map_err(|e| format!("Failed to load clip: {}", e))?;

    // Get video dimensions from the JSON (we need to read them since Clip doesn't have them yet)
    let det_content = std::fs::read_to_string(detections_path)?;
    let labeled: serde_json::Value = serde_json::from_str(&det_content)?;
    let video_width = labeled["width"].as_u64().unwrap_or(1920) as u32;
    let video_height = labeled["height"].as_u64().unwrap_or(1080) as u32;

    // Get sampled frame indices using the shared implementation
    let sampled_indices = clip.get_sampled_frame_indices(tracker_fps);

    eprintln!("Evaluating on {} frames (sampled from {})", sampled_indices.len(), clip.frame_count);
    if tracker_fps > 0.0 {
        eprintln!("  tracker_fps: {}", tracker_fps);
    }
    eprintln!("  track_thresh: {}, high_thresh: {}, track_buffer: {}s",
              track_thresh, high_thresh, track_buffer_secs);

    // Create tracker
    let mut tracker = ByteTracker::new(
        track_buffer_secs,
        track_thresh,
        high_thresh,
        false,                       // use_ciou
        1.0, 0.5,                    // high_conf_match: iou_weight, min_iou
        1.0, 0.5,                    // low_conf_match: iou_weight, min_iou
        1.0, 0.3,                    // track_activation: iou_weight, min_iou
        1.0 / 20.0,                  // kalman_std_weight_pos
        1.0 / 160.0,                 // kalman_std_weight_vel
        1.0 / 20.0,                  // kalman_std_weight_position_meas
        30.0_f32.sqrt() / 20.0,      // kalman_std_weight_position_mot
        30.0_f32.sqrt() / 160.0,     // kalman_std_weight_velocity_mot
        1e-2,                        // kalman_std_aspect_ratio_init
        1e-5,                        // kalman_std_d_aspect_ratio_init
        30.0_f32.sqrt() * 1e-2,      // kalman_std_aspect_ratio_mot
        30.0_f32.sqrt() * 1e-5,      // kalman_std_d_aspect_ratio_mot
        1e-1,                        // kalman_std_aspect_ratio_meas
    );

    // Run tracker and collect results
    let mut evaluator = Evaluator::new(iou_threshold);

    for &frame_idx in &sampled_indices {
        let timestamp_ms = clip.get_timestamp(frame_idx)
            .unwrap_or(frame_idx as u64 * 100); // fallback ~10fps

        let frame_dets = clip.get_detections(frame_idx);

        // Build tracker input objects
        let objects: Vec<Object> = frame_dets.iter()
            .enumerate()
            .map(|(i, det)| {
                let (x, y, w, h) = det.to_pixel_rect(video_width, video_height);
                Object::new(i as i64, Rect::new(x, y, w, h), det.confidence, None, None)
            })
            .collect();

        // Run tracker
        let tracked = tracker.update_with_timestamp(objects.into_iter(), timestamp_ms)
            .unwrap_or_default();

        // Collect ground truth for this frame
        let gt_dets: Vec<GtDetection> = frame_dets.iter()
            .filter_map(|det| {
                let gt_id = det.gt_track_id?;
                let (x, y, w, h) = det.to_pixel_rect(video_width, video_height);
                Some(GtDetection { frame_idx, track_id: gt_id, bbox: (x, y, w, h) })
            })
            .collect();

        // Collect predictions for this frame
        let pred_dets: Vec<PredDetection> = tracked.iter()
            .map(|obj| {
                let rect = obj.get_rect();
                PredDetection {
                    frame_idx,
                    track_id: obj.get_track_id().unwrap_or(0) as u64,
                    bbox: (rect.x(), rect.y(), rect.width(), rect.height()),
                }
            })
            .collect();

        evaluator.add_ground_truth(gt_dets);
        evaluator.add_predictions(pred_dets);
        evaluator.evaluate_frame(frame_idx);
    }

    // Compute final metrics
    evaluator.compute_hota();
    let metrics = evaluator.metrics();

    // Print results
    println!();
    println!("=== Evaluation Results ===");
    println!("MOTA:      {:.2}%", metrics.mota() * 100.0);
    println!("HOTA:      {:.2}%", metrics.hota() * 100.0);
    println!("IDF1:      {:.2}%", metrics.idf1() * 100.0);
    println!("Precision: {:.2}%", metrics.precision() * 100.0);
    println!("Recall:    {:.2}%", metrics.recall() * 100.0);
    println!();
    println!("TP:        {}", metrics.true_positives);
    println!("FP:        {}", metrics.false_positives);
    println!("FN:        {}", metrics.false_negatives);
    println!("ID Sw:     {}", metrics.id_switches);
    println!("Total GT:  {}", metrics.total_gt);

    Ok(())
}
