//! CLI evaluation script for comparing tracker results to ground truth.
//!
//! Usage:
//!   # Single file pair:
//!   cargo run --bin evaluate --release -- \
//!     <detections.json> <timestamps.json> [options]
//!
//!   # Glob patterns (matched by wildcard):
//!   cargo run --bin evaluate --release -- \
//!     "human_labeled_*.json" "timestamps_*.json" --tracker-fps 10,5
//!
//!   # With config file:
//!   cargo run --bin evaluate --release -- \
//!     <detections.json> <timestamps.json> --config bytetrack_config.yaml

use std::env;
use std::path::{Path, PathBuf};

use glob::glob;
use serde::Deserialize;
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::dataset::Clip;
use jamtrack_rs::evaluation::{Evaluator, FrameAssociation, compute_rms};
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

/// ByteTrack configuration loaded from YAML
#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct ByteTrackConfig {
    detection_min_confidence: Option<f32>,
    track_thresh: Option<f32>,
    high_thresh: Option<f32>,
    track_buffer: Option<f32>,
    use_ciou: Option<bool>,
    high_match_iou: Option<f32>,
    high_match_weight: Option<f32>,
    low_match_iou: Option<f32>,
    low_match_weight: Option<f32>,
    activate_iou: Option<f32>,
    activate_weight: Option<f32>,
    kalman_pos: Option<f32>,
    kalman_vel: Option<f32>,
    kalman_pos_meas: Option<f32>,
    kalman_pos_mot: Option<f32>,
    kalman_vel_mot: Option<f32>,
    kalman_ar_init: Option<f32>,
    kalman_dar_init: Option<f32>,
    kalman_ar_mot: Option<f32>,
    kalman_dar_mot: Option<f32>,
    kalman_ar_meas: Option<f32>,
}

fn print_usage() {
    eprintln!("Usage: evaluate <det_pattern> <ts_pattern> [options]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <det_pattern>            Detection files glob pattern (e.g., 'human_labeled_*.json')");
    eprintln!("  <ts_pattern>             Timestamps files glob pattern (e.g., 'timestamps_*.json')");
    eprintln!();
    eprintln!("Files are matched by the wildcard portion. For example:");
    eprintln!("  'human_labeled_*.json' + 'timestamps_*.json'");
    eprintln!("  matches human_labeled_foo.json with timestamps_foo.json");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --config <path>          Load ByteTrack params from YAML config file");
    eprintln!("  --tracker-fps <fps>      Target FPS, comma-separated for multiple (0 = native) [default: 0]");
    eprintln!("  --min-conf <conf>        Filter detections below this confidence [default: 0.0]");
    eprintln!("  --track-thresh <thresh>  Min confidence for low-conf pool [default: 0.25]");
    eprintln!("  --high-thresh <thresh>   Min confidence to spawn new track [default: 0.5]");
    eprintln!("  --track-buffer <secs>    Track buffer in seconds [default: 1.0]");
    eprintln!("  --use-ciou               Use CIoU instead of IoU for matching");
    eprintln!();
    eprintln!("Matching IoU thresholds:");
    eprintln!("  --high-match-iou <v>     High-conf match min IoU [default: 0.5]");
    eprintln!("  --high-match-weight <v>  High-conf match IoU weight [default: 1.0]");
    eprintln!("  --low-match-iou <v>      Low-conf match min IoU [default: 0.5]");
    eprintln!("  --low-match-weight <v>   Low-conf match IoU weight [default: 1.0]");
    eprintln!("  --activate-iou <v>       Track activation min IoU [default: 0.3]");
    eprintln!("  --activate-weight <v>    Track activation IoU weight [default: 1.0]");
    eprintln!();
    eprintln!("Kalman filter params (per-frame):");
    eprintln!("  --kalman-pos <v>         Std weight position [default: 0.05]");
    eprintln!("  --kalman-vel <v>         Std weight velocity [default: 0.00625]");
    eprintln!("  --kalman-pos-meas <v>    Std weight position measurement [default: 0.05]");
    eprintln!("  --kalman-pos-mot <v>     Std weight position motion [default: 0.05]");
    eprintln!("  --kalman-vel-mot <v>     Std weight velocity motion [default: 0.00625]");
    eprintln!("  --kalman-ar-init <v>     Std aspect ratio init [default: 0.01]");
    eprintln!("  --kalman-dar-init <v>    Std d_aspect ratio init [default: 1e-5]");
    eprintln!("  --kalman-ar-mot <v>      Std aspect ratio motion [default: 0.01]");
    eprintln!("  --kalman-dar-mot <v>     Std d_aspect ratio motion [default: 1e-5]");
    eprintln!("  --kalman-ar-meas <v>     Std aspect ratio measurement [default: 0.1]");
    eprintln!();
    eprintln!("CLI flags override config file values.");
}

/// Extract the wildcard-matched portion from a path given a pattern.
/// E.g., pattern="human_labeled_*.json", path="human_labeled_foo.json" -> Some("foo")
fn extract_wildcard_match(pattern: &str, path: &Path) -> Option<String> {
    let filename = path.file_name()?.to_str()?;
    let pattern_filename = Path::new(pattern).file_name()?.to_str()?;

    // Find the position of '*' in the pattern
    let star_pos = pattern_filename.find('*')?;
    let prefix = &pattern_filename[..star_pos];
    let suffix = &pattern_filename[star_pos + 1..];

    // Extract the matched portion from the filename
    if filename.starts_with(prefix) && filename.ends_with(suffix) {
        let matched = &filename[prefix.len()..filename.len() - suffix.len()];
        Some(matched.to_string())
    } else {
        None
    }
}

/// Build a concrete path from a pattern and a wildcard match.
/// E.g., pattern="timestamps_*.json", matched="foo" -> "timestamps_foo.json" (with directory)
fn build_path_from_pattern(pattern: &str, matched: &str) -> PathBuf {
    let pattern_path = Path::new(pattern);
    let dir = pattern_path.parent().unwrap_or(Path::new("."));
    let filename_pattern = pattern_path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("*");

    let filename = filename_pattern.replace('*', matched);
    dir.join(filename)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let det_pattern = &args[1];
    let ts_pattern = &args[2];

    // First pass: look for --config to load base values
    let mut config = ByteTrackConfig::default();
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--config" {
            i += 1;
            if let Some(config_path) = args.get(i) {
                let config_content = std::fs::read_to_string(config_path)
                    .unwrap_or_else(|e| {
                        eprintln!("Error reading config file '{}': {}", config_path, e);
                        std::process::exit(1);
                    });
                config = serde_yaml::from_str(&config_content)
                    .unwrap_or_else(|e| {
                        eprintln!("Error parsing config file '{}': {}", config_path, e);
                        std::process::exit(1);
                    });
                eprintln!("Loaded config from: {}", config_path);
            }
            break;
        }
        i += 1;
    }

    // Initialize with config values or defaults
    let mut tracker_fps_values: Vec<f32> = vec![0.0];  // default: native FPS
    let mut detection_min_conf = config.detection_min_confidence.unwrap_or(0.0);
    let mut track_thresh = config.track_thresh.unwrap_or(0.25);
    let mut high_thresh = config.high_thresh.unwrap_or(0.5);
    let mut track_buffer_secs = config.track_buffer.unwrap_or(1.0);
    let mut use_ciou = config.use_ciou.unwrap_or(false);

    // Matching IoU params
    let mut high_match_iou = config.high_match_iou.unwrap_or(0.5);
    let mut high_match_weight = config.high_match_weight.unwrap_or(1.0);
    let mut low_match_iou = config.low_match_iou.unwrap_or(0.5);
    let mut low_match_weight = config.low_match_weight.unwrap_or(1.0);
    let mut activate_iou = config.activate_iou.unwrap_or(0.3);
    let mut activate_weight = config.activate_weight.unwrap_or(1.0);

    // Kalman filter params (per-frame, no sqrt(30) scaling)
    let mut kalman_pos = config.kalman_pos.unwrap_or(1.0 / 20.0);
    let mut kalman_vel = config.kalman_vel.unwrap_or(1.0 / 160.0);
    let mut kalman_pos_meas = config.kalman_pos_meas.unwrap_or(1.0 / 20.0);
    let mut kalman_pos_mot = config.kalman_pos_mot.unwrap_or(1.0 / 20.0);
    let mut kalman_vel_mot = config.kalman_vel_mot.unwrap_or(1.0 / 160.0);
    let mut kalman_ar_init = config.kalman_ar_init.unwrap_or(1e-2);
    let mut kalman_dar_init = config.kalman_dar_init.unwrap_or(1e-5);
    let mut kalman_ar_mot = config.kalman_ar_mot.unwrap_or(1e-2);
    let mut kalman_dar_mot = config.kalman_dar_mot.unwrap_or(1e-5);
    let mut kalman_ar_meas = config.kalman_ar_meas.unwrap_or(1e-1);

    // Second pass: CLI flags override config values
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1; // Skip config path, already processed
            }
            "--tracker-fps" => {
                i += 1;
                if let Some(fps_str) = args.get(i) {
                    tracker_fps_values = fps_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if tracker_fps_values.is_empty() {
                        tracker_fps_values = vec![0.0];
                    }
                }
            }
            "--min-conf" => {
                i += 1;
                detection_min_conf = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
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
            "--use-ciou" => {
                use_ciou = true;
            }
            "--high-match-iou" => {
                i += 1;
                high_match_iou = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            }
            "--high-match-weight" => {
                i += 1;
                high_match_weight = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            "--low-match-iou" => {
                i += 1;
                low_match_iou = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            }
            "--low-match-weight" => {
                i += 1;
                low_match_weight = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            "--activate-iou" => {
                i += 1;
                activate_iou = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.3);
            }
            "--activate-weight" => {
                i += 1;
                activate_weight = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            "--kalman-pos" => {
                i += 1;
                kalman_pos = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0 / 20.0);
            }
            "--kalman-vel" => {
                i += 1;
                kalman_vel = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0 / 160.0);
            }
            "--kalman-pos-meas" => {
                i += 1;
                kalman_pos_meas = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0 / 20.0);
            }
            "--kalman-pos-mot" => {
                i += 1;
                kalman_pos_mot = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0 / 20.0);
            }
            "--kalman-vel-mot" => {
                i += 1;
                kalman_vel_mot = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0 / 160.0);
            }
            "--kalman-ar-init" => {
                i += 1;
                kalman_ar_init = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1e-2);
            }
            "--kalman-dar-init" => {
                i += 1;
                kalman_dar_init = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1e-5);
            }
            "--kalman-ar-mot" => {
                i += 1;
                kalman_ar_mot = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1e-2);
            }
            "--kalman-dar-mot" => {
                i += 1;
                kalman_dar_mot = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1e-5);
            }
            "--kalman-ar-meas" => {
                i += 1;
                kalman_ar_meas = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1e-1);
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

    // Collect file pairs
    let file_pairs: Vec<(PathBuf, PathBuf)> = if det_pattern.contains('*') {
        // Glob pattern mode - match files by wildcard
        let mut pairs = Vec::new();
        for entry in glob(det_pattern)?.flatten() {
            if let Some(matched) = extract_wildcard_match(det_pattern, &entry) {
                let ts_path = build_path_from_pattern(ts_pattern, &matched);
                if ts_path.exists() {
                    pairs.push((entry, ts_path));
                } else {
                    eprintln!("Warning: No timestamps file for {:?} (expected {:?})", entry, ts_path);
                }
            }
        }
        pairs.sort();
        pairs
    } else {
        // Single file mode
        let det_path = PathBuf::from(det_pattern);
        let ts_path = PathBuf::from(ts_pattern);
        if !det_path.exists() {
            eprintln!("Error: Detection file not found: {}", det_pattern);
            std::process::exit(1);
        }
        if !ts_path.exists() {
            eprintln!("Error: Timestamps file not found: {}", ts_pattern);
            std::process::exit(1);
        }
        vec![(det_path, ts_path)]
    };

    if file_pairs.is_empty() {
        eprintln!("No matching file pairs found");
        std::process::exit(1);
    }

    eprintln!("Found {} file pair(s)", file_pairs.len());
    eprintln!("FPS values to evaluate: {:?}", tracker_fps_values);
    eprintln!();

    // Aggregated stats per FPS setting
    struct AggregateStats {
        matched: usize,
        false_match: usize,
        untracked: usize,
        tracked_no_object: usize,
        total_switches: Vec<usize>,
        bad_switches: Vec<usize>,
        total_switches_sum: usize,
        total_bad_switches_sum: usize,
        total_duration_secs: f64,
        total_frames: usize,
        num_sequences: usize,
    }

    impl AggregateStats {
        fn new() -> Self {
            Self {
                matched: 0,
                false_match: 0,
                untracked: 0,
                tracked_no_object: 0,
                total_switches: Vec::new(),
                bad_switches: Vec::new(),
                total_switches_sum: 0,
                total_bad_switches_sum: 0,
                total_duration_secs: 0.0,
                total_frames: 0,
                num_sequences: 0,
            }
        }
    }

    let mut aggregate_by_fps: std::collections::HashMap<String, AggregateStats> = std::collections::HashMap::new();
    for &fps in &tracker_fps_values {
        let label = if fps == 0.0 { "native".to_string() } else { format!("{}", fps) };
        aggregate_by_fps.insert(label, AggregateStats::new());
    }

    // Process each file pair
    for (det_path, timestamps_path) in &file_pairs {
        let video_name = det_path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        println!("========================================");
        println!("Video: {}", video_name);
        println!("  Detections:  {:?}", det_path);
        println!("  Timestamps:  {:?}", timestamps_path);

        // Load clip
        let mut clip = match Clip::load_human_labeled(det_path, &timestamps_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Error loading clip: {}", e);
                continue;
            }
        };

        // Resolve duplicate GT IDs (same GT ID appearing multiple times in a frame)
        // Keeps only the longest track, nullifies the rest
        let (num_dups, num_nullified) = clip.resolve_duplicate_gt_ids();
        if num_dups > 0 {
            println!("  Resolved {} duplicate GT IDs ({} detections nullified)", num_dups, num_nullified);
        }

        // Get video dimensions from the JSON
        let det_content = std::fs::read_to_string(det_path)?;
        let labeled: serde_json::Value = serde_json::from_str(&det_content)?;
        let video_width = labeled["width"].as_u64().unwrap_or(1920) as u32;
        let video_height = labeled["height"].as_u64().unwrap_or(1080) as u32;

        // Run evaluation at each FPS
        for &tracker_fps in &tracker_fps_values {
            let fps_label = if tracker_fps == 0.0 {
                "native".to_string()
            } else {
                format!("{}", tracker_fps)
            };

            println!();
            println!("--- {} FPS ---", fps_label);

            let sampled_indices = clip.get_sampled_frame_indices(tracker_fps);
            eprintln!("  Evaluating {} frames (sampled from {})",
                     sampled_indices.len(), clip.frame_count);

            // Create fresh tracker for each FPS setting
            let mut tracker = ByteTracker::new(
                30,                          // frame_rate (nominal)
                track_buffer_secs,
                track_thresh,
                high_thresh,
                use_ciou,
                high_match_weight, high_match_iou,
                low_match_weight, low_match_iou,
                activate_weight, activate_iou,
                kalman_pos,
                kalman_vel,
                kalman_pos_meas,
                kalman_pos_mot,
                kalman_vel_mot,
                kalman_ar_init,
                kalman_dar_init,
                kalman_ar_mot,
                kalman_dar_mot,
                kalman_ar_meas,
            );

            let mut evaluator = Evaluator::new();

            for &frame_idx in &sampled_indices {
                let timestamp_ms = clip.get_timestamp(frame_idx)
                    .unwrap_or(frame_idx as u64 * 100);

                let frame_dets: Vec<_> = clip.get_detections(frame_idx)
                    .iter()
                    .filter(|det| det.confidence >= detection_min_conf)
                    .cloned()
                    .collect();

                let objects: Vec<Object> = frame_dets.iter()
                    .enumerate()
                    .map(|(i, det)| {
                        let (x, y, w, h) = det.to_pixel_rect(video_width, video_height);
                        Object::new(i as i64, Rect::new(x, y, w, h), det.confidence, None, None)
                    })
                    .collect();

                let tracked = tracker.update_with_timestamp(objects.into_iter(), timestamp_ms)
                    .unwrap_or_default();

                // Each tracker output corresponds to an input detection (same boxes, just with track IDs)
                // Use detection_id to look up the original gt_track_id
                for obj in &tracked {
                    let detection_id = obj.get_detection_id() as usize;
                    let tracker_id = obj.get_track_id().unwrap_or(0) as u64;

                    let gt_track_id = frame_dets.get(detection_id)
                        .and_then(|det| det.gt_track_id);

                    evaluator.add_association(FrameAssociation {
                        frame_idx,
                        tracker_id,
                        gt_track_id,
                    });
                }
            }

            let result = evaluator.compute_association_score();
            let metrics = &result.metrics;

            // Compute duration from timestamps
            let first_ts = sampled_indices.first()
                .and_then(|&idx| clip.get_timestamp(idx))
                .unwrap_or(0);
            let last_ts = sampled_indices.last()
                .and_then(|&idx| clip.get_timestamp(idx))
                .unwrap_or(0);
            let duration_secs = (last_ts - first_ts) as f64 / 1000.0;
            let num_frames = sampled_indices.len();

            // Normalized metrics
            let frag_per_sec = if duration_secs > 0.0 { metrics.total_switches as f64 / duration_secs } else { 0.0 };
            let frag_per_frame = if num_frames > 0 { metrics.total_switches as f64 / num_frames as f64 } else { 0.0 };
            let conf_per_sec = if duration_secs > 0.0 { metrics.total_bad_switches as f64 / duration_secs } else { 0.0 };
            let conf_per_frame = if num_frames > 0 { metrics.total_bad_switches as f64 / num_frames as f64 } else { 0.0 };

            println!("  Coverage: {:.2}%", metrics.coverage() * 100.0);
            println!("  Fragmentation: {:.2} (RMS), {:.3}/sec, {:.4}/frame",
                    metrics.fragmentation, frag_per_sec, frag_per_frame);
            println!("  Confusion: {:.2} (RMS), {:.3}/sec, {:.4}/frame",
                    metrics.confusion, conf_per_sec, conf_per_frame);
            println!("  Tracked No Object: {}", metrics.tracked_no_object);

            // Update aggregates
            if let Some(agg) = aggregate_by_fps.get_mut(&fps_label) {
                agg.matched += metrics.matched;
                agg.false_match += metrics.false_match;
                agg.untracked += metrics.untracked;
                agg.tracked_no_object += metrics.tracked_no_object;
                agg.total_switches.extend(&result.total_switches_per_gt);
                agg.bad_switches.extend(&result.bad_switches_per_gt);
                agg.total_switches_sum += metrics.total_switches;
                agg.total_bad_switches_sum += metrics.total_bad_switches;
                agg.total_duration_secs += duration_secs;
                agg.total_frames += num_frames;
                agg.num_sequences += 1;
            }
        }
        println!();
    }

    // Print aggregate statistics
    if file_pairs.len() > 1 {
        println!("========================================");
        println!("AGGREGATE STATISTICS ({} sequences)", file_pairs.len());
        println!("========================================");

        for &tracker_fps in &tracker_fps_values {
            let fps_label = if tracker_fps == 0.0 { "native".to_string() } else { format!("{}", tracker_fps) };

            if let Some(agg) = aggregate_by_fps.get(&fps_label) {
                println!();
                println!("--- {} FPS ---", fps_label);

                let denom = agg.matched + agg.false_match + agg.untracked;
                let coverage = if denom == 0 { 0.0 } else { agg.matched as f32 / denom as f32 };
                let fragmentation = compute_rms(&agg.total_switches);
                let confusion = compute_rms(&agg.bad_switches);

                // Normalized metrics
                let frag_per_sec = if agg.total_duration_secs > 0.0 {
                    agg.total_switches_sum as f64 / agg.total_duration_secs
                } else { 0.0 };
                let frag_per_frame = if agg.total_frames > 0 {
                    agg.total_switches_sum as f64 / agg.total_frames as f64
                } else { 0.0 };
                let conf_per_sec = if agg.total_duration_secs > 0.0 {
                    agg.total_bad_switches_sum as f64 / agg.total_duration_secs
                } else { 0.0 };
                let conf_per_frame = if agg.total_frames > 0 {
                    agg.total_bad_switches_sum as f64 / agg.total_frames as f64
                } else { 0.0 };

                println!("  Coverage: {:.2}%", coverage * 100.0);
                println!("  Fragmentation: {:.2} (RMS), {:.3}/sec, {:.4}/frame",
                        fragmentation, frag_per_sec, frag_per_frame);
                println!("  Confusion: {:.2} (RMS), {:.3}/sec, {:.4}/frame",
                        confusion, conf_per_sec, conf_per_frame);
                println!("  Tracked No Object: {}", agg.tracked_no_object);
                println!("  (Matched: {}, FalseMatch: {}, Untracked: {})",
                        agg.matched, agg.false_match, agg.untracked);
                println!("  (Total duration: {:.1}s, {} frames)", agg.total_duration_secs, agg.total_frames);
            }
        }
        println!();
    }

    println!("========================================");
    println!("Evaluation complete!");

    Ok(())
}
