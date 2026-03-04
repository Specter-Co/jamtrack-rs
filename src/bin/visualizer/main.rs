mod drawing;
mod ui;
mod video;

use jamtrack_rs::dataset::Clip;
use jamtrack_rs::evaluation::{Evaluator, FrameAssociation, AssociationResult};
use eframe::egui;
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::debug_info::FrameDebugInfo;
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;
use std::collections::HashMap;
use std::path::PathBuf;
use video::{DecoderHandle, VideoMeta};

/// Command line arguments for the visualizer
struct CliArgs {
    /// Path to video file
    video_path: Option<PathBuf>,
    /// Path to detections JSON file (with optional GT labels)
    detections_path: Option<PathBuf>,
    /// Path to timestamps JSON file
    timestamps_path: Option<PathBuf>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 4 {
        // Full format: <video> <detections.json> <timestamps.json>
        CliArgs {
            video_path: Some(PathBuf::from(&args[1])),
            detections_path: Some(PathBuf::from(&args[2])),
            timestamps_path: Some(PathBuf::from(&args[3])),
        }
    } else if args.len() == 1 {
        CliArgs {
            video_path: None,
            detections_path: None,
            timestamps_path: None,
        }
    } else {
        eprintln!("Usage: visualizer <video.mp4> <detections.json> <timestamps.json>");
        std::process::exit(1);
    }
}

fn main() -> eframe::Result<()> {
    let cli = parse_args();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_resizable(true)
            .with_title("ByteTrack Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "ByteTrack Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(VisualizerApp::new(cc, cli)))),
    )
}

/// Overlay visibility settings
#[derive(Clone)]
pub struct OverlaySettings {
    pub show_detections: bool,
    pub show_tracks: bool,
    pub show_track_ids: bool,
    pub show_confidence: bool,
    pub show_velocities: bool,
    pub show_class_labels: bool,
    pub detection_min_confidence: f32,
    // Class filter - empty means show all
    pub enabled_classes: std::collections::HashSet<String>,
    pub all_classes: Vec<String>,
    // ByteTrack mechanics visualization (animated step-by-step)
    pub show_bytetrack_mechanics: bool,
    // Evaluation overlays (only when ground truth is available)
    pub show_eval_overlays: bool,
    pub show_gt_boxes: bool,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        Self {
            show_detections: true,
            show_tracks: true,
            show_track_ids: true,
            show_confidence: false,
            show_velocities: false,
            show_class_labels: true,
            detection_min_confidence: 0.1,
            enabled_classes: std::collections::HashSet::new(), // Empty = show all
            all_classes: Vec::new(),
            // ByteTrack mechanics visualization
            show_bytetrack_mechanics: false,
            // Evaluation overlays
            show_eval_overlays: false,
            show_gt_boxes: true,
        }
    }
}

/// Tracker parameter settings
#[derive(Clone, PartialEq)]
pub struct TrackerParams {
    // Core params
    pub frame_rate: usize,
    pub track_buffer_secs: f32,
    pub track_thresh: f32,
    pub high_thresh: f32,
    pub tracker_fps: f32,  // Target FPS for tracking (0 = use all frames)

    // Matching params
    pub use_ciou: bool,
    pub high_conf_match_min_iou: f32,
    pub high_conf_match_iou_weight: f32,
    pub low_conf_match_min_iou: f32,
    pub low_conf_match_iou_weight: f32,
    pub track_activation_min_iou: f32,
    pub track_activation_iou_weight: f32,

    // Kalman filter params
    pub kalman_std_weight_pos: f32,
    pub kalman_std_weight_vel: f32,
    pub kalman_std_weight_position_meas: f32,
    pub kalman_std_weight_position_mot: f32,
    pub kalman_std_weight_velocity_mot: f32,
    pub kalman_std_aspect_ratio_init: f32,
    pub kalman_std_d_aspect_ratio_init: f32,
    pub kalman_std_aspect_ratio_mot: f32,
    pub kalman_std_d_aspect_ratio_mot: f32,
    pub kalman_std_aspect_ratio_meas: f32,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            frame_rate: 30,
            track_buffer_secs: 1.0,
            track_thresh: 0.25,
            high_thresh: 0.5,
            tracker_fps: 0.0,  // 0 means use native video FPS (all frames)

            use_ciou: false,
            high_conf_match_min_iou: 0.5,
            high_conf_match_iou_weight: 1.0,
            low_conf_match_min_iou: 0.5,
            low_conf_match_iou_weight: 1.0,
            track_activation_min_iou: 0.3,
            track_activation_iou_weight: 1.0,

            // Init and measurement params at original values
            kalman_std_weight_pos: 1.0 / 20.0,
            kalman_std_weight_vel: 1.0 / 160.0,
            kalman_std_weight_position_meas: 1.0 / 20.0,
            kalman_std_aspect_ratio_init: 1e-2,
            kalman_std_d_aspect_ratio_init: 1e-5,
            kalman_std_aspect_ratio_meas: 1e-1,
            // Motion/process noise params (scaled by √30 to match original 30fps calibration)
            kalman_std_weight_position_mot: 30.0_f32.sqrt() / 20.0,
            kalman_std_weight_velocity_mot: 30.0_f32.sqrt() / 160.0,
            kalman_std_aspect_ratio_mot: 30.0_f32.sqrt() * 1e-2,
            kalman_std_d_aspect_ratio_mot: 30.0_f32.sqrt() * 1e-5,
        }
    }
}

/// A tracked object result from ByteTrack
#[derive(Clone, Debug)]
pub struct TrackedObject {
    pub track_id: usize,
    pub rect: Rect<f32>,
    pub confidence: f32,
    pub velocity: (f32, f32),
}

/// Track results for all frames
pub struct TrackResults {
    pub results_by_frame: HashMap<usize, Vec<TrackedObject>>,
    pub debug_by_frame: HashMap<usize, FrameDebugInfo>,
}

/// Main application state
pub struct VisualizerApp {
    // Data
    clip: Option<Clip>,
    decoder: Option<DecoderHandle>,
    load_error: Option<String>,

    // Track results from ByteTrack
    track_results: Option<TrackResults>,
    last_tracker_params: TrackerParams,
    last_enabled_classes: std::collections::HashSet<String>,
    last_detection_min_confidence: f32,

    // Playback state
    current_frame: usize,
    is_playing: bool,
    playback_fps: f32,
    last_frame_time: Option<std::time::Instant>,

    // Video texture
    video_texture: Option<egui::TextureHandle>,
    displayed_original_frame: Option<usize>,  // Original frame index currently in texture
    video_width: u32,
    video_height: u32,

    // UI state
    overlay_settings: OverlaySettings,
    tracker_params: TrackerParams,

    // ByteTrack mechanics animation state
    mechanics_stage: usize,  // 0=predictions, 1=detections, 2=stage1 match, 3=stage2 match, 4=final

    // Hover state for highlighting related entities
    hovered_track_id: Option<usize>,

    // Frame sampling for lower tracker FPS (maps display index -> original frame index)
    sampled_frames: Vec<usize>,

    // Debug: track ID to inspect
    debug_track_id_input: String,
    debug_track_id: Option<usize>,

    // Evaluation state (when ground truth is available)
    association_result: Option<AssociationResult>,
    // Per-frame evaluation data: frame_idx -> Vec<(tracker_id, gt_track_id, is_correct_match)>
    frame_eval_data: HashMap<usize, Vec<(u64, Option<u64>, bool)>>,
    // Ground truth bboxes keyed by display frame index
    gt_by_display_frame: HashMap<usize, Vec<(u64, (f32, f32, f32, f32))>>,
    // GT time series: gt_id -> Vec<(display_frame_idx, Option<pred_id>)> sorted by frame
    gt_timeseries: HashMap<u64, Vec<(usize, Option<u64>)>>,

    // Pending file uploads (for separate Video/Detections/Timestamps buttons)
    pending_video: Option<PathBuf>,
    pending_detections: Option<PathBuf>,
    pending_timestamps: Option<PathBuf>,
}

impl VisualizerApp {
    fn new(_cc: &eframe::CreationContext<'_>, cli: CliArgs) -> Self {
        let tracker_params = TrackerParams::default();
        let mut app = Self {
            clip: None,
            decoder: None,
            load_error: None,
            track_results: None,
            last_tracker_params: tracker_params.clone(),
            last_enabled_classes: std::collections::HashSet::new(),
            last_detection_min_confidence: 0.1,  // Default matches OverlaySettings default
            current_frame: 0,
            is_playing: false,
            playback_fps: 5.0,
            last_frame_time: None,
            video_texture: None,
            displayed_original_frame: None,
            video_width: 0,
            video_height: 0,
            overlay_settings: OverlaySettings::default(),
            tracker_params,
            mechanics_stage: 0,
            hovered_track_id: None,
            sampled_frames: Vec::new(),
            debug_track_id_input: String::new(),
            debug_track_id: None,
            association_result: None,
            frame_eval_data: HashMap::new(),
            gt_by_display_frame: HashMap::new(),
            gt_timeseries: HashMap::new(),
            pending_video: None,
            pending_detections: None,
            pending_timestamps: None,
        };

        // Load from command line arguments
        if let (Some(video), Some(det_path), Some(ts_path)) = (cli.video_path, cli.detections_path, cli.timestamps_path) {
            app.load_with_video(&video, &det_path, &ts_path);
        }

        app
    }

    /// Load a dataset with explicit video path
    pub fn load_with_video(&mut self, video_path: &std::path::Path, json_path: &std::path::Path, timestamps_path: &std::path::Path) {
        // Reset state
        self.clip = None;
        self.decoder = None;
        self.load_error = None;
        self.track_results = None;
        self.current_frame = 0;
        self.video_texture = None;
        self.sampled_frames = Vec::new();
        self.association_result = None;
        self.frame_eval_data = HashMap::new();
        self.gt_by_display_frame = HashMap::new();
        self.gt_timeseries = HashMap::new();

        match Clip::load_human_labeled(json_path, timestamps_path) {
            Ok(mut clip) => {
                // Resolve duplicate GT IDs (same GT ID appearing multiple times in a frame)
                // Keeps only the longest track, nullifies the rest
                let (num_dups, num_nullified) = clip.resolve_duplicate_gt_ids();
                if num_dups > 0 {
                    eprintln!("Resolved {} duplicate GT IDs ({} detections nullified)", num_dups, num_nullified);
                }

                // Override video path from command line
                clip.video_path = video_path.to_path_buf();
                let video_path_str = video_path.to_string_lossy().to_string();
                eprintln!("Opening video: {}", video_path_str);

                match VideoMeta::open(&video_path_str, 960, 540) {
                    Some(meta) => {
                        clip.set_video_dimensions(meta.src_width, meta.src_height);
                        self.video_width = meta.out_width;
                        self.video_height = meta.out_height;

                        let decoder = DecoderHandle::spawn(meta);
                        self.decoder = Some(decoder);
                        self.clip = Some(clip);

                        eprintln!("Loaded clip with {} frames (has_ground_truth={})",
                            self.clip.as_ref().unwrap().frame_count,
                            self.clip.as_ref().unwrap().has_ground_truth);

                        // Collect unique classes
                        let mut classes: std::collections::HashSet<String> = std::collections::HashSet::new();
                        for frame_idx in 0..self.clip.as_ref().unwrap().frame_count {
                            for det in self.clip.as_ref().unwrap().get_detections(frame_idx) {
                                classes.insert(det.class.clone());
                            }
                        }
                        let mut class_list: Vec<String> = classes.iter().cloned().collect();
                        class_list.sort();
                        self.overlay_settings.all_classes = class_list;
                        self.overlay_settings.enabled_classes = classes;

                        // Run tracker
                        self.run_tracker();
                    }
                    None => {
                        self.load_error = Some(format!("Failed to open video: {}", video_path_str));
                    }
                }
            }
            Err(e) => {
                self.load_error = Some(e);
            }
        }
    }

    /// Run ByteTrack on all detections
    pub fn run_tracker(&mut self) {
        let clip = match &self.clip {
            Some(c) => c,
            None => return,
        };

        // Get sampled frame indices based on tracker_fps
        let sampled_indices = clip.get_sampled_frame_indices(self.tracker_params.tracker_fps);
        self.sampled_frames = sampled_indices.clone();

        let params = &self.tracker_params;

        // Create ByteTracker with current parameters
        let mut tracker = ByteTracker::new(
            params.frame_rate,
            params.track_buffer_secs,
            params.track_thresh,
            params.high_thresh,
            params.use_ciou,
            params.high_conf_match_iou_weight,
            params.high_conf_match_min_iou,
            params.low_conf_match_iou_weight,
            params.low_conf_match_min_iou,
            params.track_activation_iou_weight,
            params.track_activation_min_iou,
            params.kalman_std_weight_pos,
            params.kalman_std_weight_vel,
            params.kalman_std_weight_position_meas,
            params.kalman_std_weight_position_mot,
            params.kalman_std_weight_velocity_mot,
            params.kalman_std_aspect_ratio_init,
            params.kalman_std_d_aspect_ratio_init,
            params.kalman_std_aspect_ratio_mot,
            params.kalman_std_d_aspect_ratio_mot,
            params.kalman_std_aspect_ratio_meas,
        );

        let mut results_by_frame: HashMap<usize, Vec<TrackedObject>> = HashMap::new();
        let mut debug_by_frame: HashMap<usize, FrameDebugInfo> = HashMap::new();

        // Process only sampled frames
        for (display_idx, &frame_idx) in sampled_indices.iter().enumerate() {
            let detections = clip.get_detections(frame_idx);

            // Get timestamp in milliseconds for this frame (u64 to avoid precision loss)
            let timestamp_ms: u64 = clip.get_timestamp(frame_idx)
                .unwrap_or((display_idx as u64 * 1000) / params.frame_rate as u64);

            // Convert detections to Object format, filtering by enabled classes
            // Use pixel coordinates (ByteTrack expects pixel coords)
            let enabled_classes = &self.overlay_settings.enabled_classes;
            let objects: Vec<Object> = detections
                .iter()
                .enumerate()
                .filter(|(_, det)| {
                    // If no classes enabled, include all; otherwise filter
                    enabled_classes.is_empty() || enabled_classes.contains(&det.class)
                })
                .map(|(i, det)| {
                    let (x, y, w, h) = det.to_pixel_rect(clip.video_width, clip.video_height);
                    Object::new(
                        i as i64,
                        Rect::new(x, y, w, h),
                        det.confidence,
                        None,
                        None,
                    )
                })
                .collect();

            // Run tracker update with debug info and timestamp (u64 ms)
            let (tracked, debug_info) = tracker
                .update_with_debug_timestamp(objects.into_iter(), timestamp_ms)
                .unwrap_or_default();

            // Store results keyed by display index (not original frame index)
            let frame_results: Vec<TrackedObject> = tracked
                .into_iter()
                .map(|obj| TrackedObject {
                    track_id: obj.get_track_id().unwrap_or(0),
                    rect: obj.get_rect(),
                    confidence: obj.get_prob(),
                    velocity: obj.get_track_vel_xy().unwrap_or((0.0, 0.0)),
                })
                .collect();

            results_by_frame.insert(display_idx, frame_results);
            debug_by_frame.insert(display_idx, debug_info);
        }

        // Save frame_count before dropping the clip borrow
        let total_frame_count = clip.frame_count;

        self.track_results = Some(TrackResults {
            results_by_frame: results_by_frame.clone(),
            debug_by_frame,
        });
        self.last_tracker_params = self.tracker_params.clone();
        self.last_enabled_classes = self.overlay_settings.enabled_classes.clone();
        self.last_detection_min_confidence = self.overlay_settings.detection_min_confidence;

        // Run evaluation if ground truth is available
        self.run_evaluation(&results_by_frame, &sampled_indices);

        // Reset all playback/video state after re-running tracker
        self.current_frame = 0;
        self.mechanics_stage = 0;
        self.video_texture = None;
        self.displayed_original_frame = None;
        self.hovered_track_id = None;
        self.is_playing = false;
        self.last_frame_time = None;

        eprintln!("Tracker finished processing {} sampled frames (from {} total)",
            sampled_indices.len(), total_frame_count);
    }

    /// Run evaluation comparing tracker results to ground truth
    fn run_evaluation(
        &mut self,
        results_by_frame: &HashMap<usize, Vec<TrackedObject>>,
        sampled_indices: &[usize],
    ) {
        let clip = match &self.clip {
            Some(c) if c.has_ground_truth => c,
            _ => {
                self.association_result = None;
                self.frame_eval_data = HashMap::new();
                self.gt_by_display_frame = HashMap::new();
                self.gt_timeseries = HashMap::new();
                return;
            }
        };

        let mut evaluator = Evaluator::new();
        let mut gt_by_display_frame: HashMap<usize, Vec<(u64, (f32, f32, f32, f32))>> = HashMap::new();
        // Temporary storage for per-frame associations before we have the final assignment
        let mut frame_associations: HashMap<usize, Vec<(u64, Option<u64>)>> = HashMap::new();

        // Track deduplication stats
        let mut total_duplicates_removed = 0usize;
        let mut dedup_log: Vec<(usize, u64)> = Vec::new(); // (frame_idx, gt_id) for pruned duplicates

        // Collect ground truth bboxes and match tracker outputs to detections
        for (display_idx, &frame_idx) in sampled_indices.iter().enumerate() {
            let detections = clip.get_detections(frame_idx);
            let tracks = results_by_frame.get(&display_idx);

            // Deduplicate: group detections by gt_track_id, keep best IoU match with tracker
            // For each gt_id, find the detection that best matches a tracker output
            let mut gt_id_to_best_det: HashMap<u64, (usize, f32, (f32, f32, f32, f32))> = HashMap::new();
            // Maps gt_id -> (det_index, best_iou_with_any_tracker, pixel_rect)

            for (det_idx, det) in detections.iter().enumerate() {
                if let Some(gt_id) = det.gt_track_id {
                    let (x, y, w, h) = det.to_pixel_rect(clip.video_width, clip.video_height);
                    let det_rect = Rect::new(x, y, w, h);

                    // Find best IoU with any tracker output
                    let best_iou = tracks.map(|ts| {
                        ts.iter()
                            .map(|t| t.rect.calc_iou(&det_rect))
                            .fold(0.0f32, f32::max)
                    }).unwrap_or(0.0);

                    // Keep this detection if it has better IoU than existing one for this gt_id
                    let dominated = gt_id_to_best_det.get(&gt_id)
                        .map(|(_, prev_iou, _)| *prev_iou >= best_iou)
                        .unwrap_or(false);

                    if !dominated {
                        if gt_id_to_best_det.contains_key(&gt_id) {
                            total_duplicates_removed += 1;
                            dedup_log.push((frame_idx, gt_id));
                        }
                        gt_id_to_best_det.insert(gt_id, (det_idx, best_iou, (x, y, w, h)));
                    } else {
                        total_duplicates_removed += 1;
                        dedup_log.push((frame_idx, gt_id));
                    }
                }
            }

            // Collect deduplicated GT bboxes for visualization
            let gt_bboxes: Vec<(u64, (f32, f32, f32, f32))> = gt_id_to_best_det
                .iter()
                .map(|(&gt_id, &(_, _, rect))| (gt_id, rect))
                .collect();
            gt_by_display_frame.insert(display_idx, gt_bboxes);

            // Build set of valid detection indices (after deduplication)
            let valid_det_indices: std::collections::HashSet<usize> = gt_id_to_best_det
                .values()
                .map(|(idx, _, _)| *idx)
                .collect();

            // Match each tracker output to input detections by IoU (only considering deduplicated dets)
            let mut frame_assocs = Vec::new();
            if let Some(track_list) = tracks {
                for track in track_list {
                    let tracker_rect = &track.rect;
                    let tracker_id = track.track_id as u64;

                    // Find best matching detection by IoU (only from valid deduplicated set)
                    let mut best_iou = 0.0f32;
                    let mut best_gt_id: Option<u64> = None;

                    for (det_idx, det) in detections.iter().enumerate() {
                        // Skip detections that were deduplicated away
                        if det.gt_track_id.is_some() && !valid_det_indices.contains(&det_idx) {
                            continue;
                        }

                        let (dx, dy, dw, dh) = det.to_pixel_rect(clip.video_width, clip.video_height);
                        let det_rect = Rect::new(dx, dy, dw, dh);
                        let iou = tracker_rect.calc_iou(&det_rect);

                        if iou > best_iou {
                            best_iou = iou;
                            best_gt_id = det.gt_track_id;
                        }
                    }

                    // Only add association if we found a matching detection
                    if best_iou > 0.3 {
                        evaluator.add_association(FrameAssociation {
                            frame_idx: display_idx,
                            tracker_id,
                            gt_track_id: best_gt_id,
                        });
                        frame_assocs.push((tracker_id, best_gt_id));
                    }
                }
            }
            frame_associations.insert(display_idx, frame_assocs);
        }

        if total_duplicates_removed > 0 {
            eprintln!("[WARN] Deduplication: removed {} duplicate GT annotations (same GT ID in same frame)",
                total_duplicates_removed);
            for (frame_idx, gt_id) in &dedup_log {
                eprintln!("[WARN]   Frame {}: pruned duplicate GT ID {}", frame_idx, gt_id);
            }
        }

        // Compute Association Score metrics
        let result = evaluator.compute_association_score();
        let metrics = &result.metrics;

        eprintln!("Evaluation complete:");
        eprintln!("  Coverage: {:.2}%", metrics.coverage() * 100.0);
        eprintln!("  Fragmentation: {:.2}", metrics.fragmentation);
        eprintln!("  Confusion: {:.2}", metrics.confusion);
        eprintln!("  Tracked No Object: {}", metrics.tracked_no_object);

        // Build per-frame evaluation data with is_correct_match flag
        let mut frame_eval_data: HashMap<usize, Vec<(u64, Option<u64>, bool)>> = HashMap::new();
        for (display_idx, assocs) in frame_associations {
            let eval_data: Vec<(u64, Option<u64>, bool)> = assocs.iter().map(|(tracker_id, gt_track_id)| {
                // Check if this is a correct match based on the optimal assignment
                let is_correct = match (gt_track_id, result.assignment.get(tracker_id)) {
                    (Some(gt_id), Some(assigned_gt)) => gt_id == assigned_gt,
                    _ => false,
                };
                (*tracker_id, *gt_track_id, is_correct)
            }).collect();
            frame_eval_data.insert(display_idx, eval_data);
        }

        // Build pred_id time series: for each pred_id, track which GT it's tracking per frame
        // pred_id -> Vec<(frame_idx, Option<gt_id>)>
        let mut pred_timeseries: HashMap<u64, Vec<(usize, Option<u64>)>> = HashMap::new();

        // Build from frame_eval_data
        for (&display_idx, eval_data) in &frame_eval_data {
            for &(tracker_id, gt_id, _) in eval_data {
                pred_timeseries
                    .entry(tracker_id)
                    .or_default()
                    .push((display_idx, gt_id));
            }
        }

        // Sort each pred_id's time series by frame
        for series in pred_timeseries.values_mut() {
            series.sort_by_key(|(frame, _)| *frame);
        }

        // Also collect all unique GT IDs for Y-axis scale
        let mut all_gt_ids: Vec<u64> = gt_by_display_frame
            .values()
            .flat_map(|boxes| boxes.iter().map(|(gt_id, _)| *gt_id))
            .collect();
        all_gt_ids.sort();
        all_gt_ids.dedup();

        // Store gt_timeseries as pred_timeseries (rename the field usage)
        let gt_timeseries = pred_timeseries;

        self.association_result = Some(result);
        self.frame_eval_data = frame_eval_data;
        self.gt_by_display_frame = gt_by_display_frame;
        self.gt_timeseries = gt_timeseries;
        self.overlay_settings.show_eval_overlays = true;  // Auto-enable when GT is available
    }

    /// Check if tracker params or class filter changed and re-run if needed
    pub fn check_rerun_tracker(&mut self) {
        let params_changed = self.tracker_params != self.last_tracker_params;
        let classes_changed = self.overlay_settings.enabled_classes != self.last_enabled_classes;
        if (params_changed || classes_changed) && self.clip.is_some() {
            self.run_tracker();
        }
    }

    fn frame_count(&self) -> usize {
        if self.sampled_frames.is_empty() {
            self.clip.as_ref().map(|c| c.frame_count).unwrap_or(0)
        } else {
            self.sampled_frames.len()
        }
    }

    /// Get the original frame index for a display frame index
    fn get_original_frame_idx(&self, display_idx: usize) -> usize {
        if self.sampled_frames.is_empty() {
            display_idx
        } else {
            self.sampled_frames.get(display_idx).copied().unwrap_or(0)
        }
    }

    /// Get the display frame index for an original frame index (reverse lookup)
    /// Returns None if the original frame is not in the sampled set
    fn get_display_idx_for_original(&self, original_idx: usize) -> Option<usize> {
        if self.sampled_frames.is_empty() {
            Some(original_idx)
        } else {
            self.sampled_frames.iter().position(|&idx| idx == original_idx)
        }
    }

    /// Get association evaluation result
    pub fn get_association_result(&self) -> Option<&AssociationResult> {
        self.association_result.as_ref()
    }

    /// Get frame evaluation data for the current frame
    /// Returns Vec<(tracker_id, gt_track_id, is_correct_match)>
    pub fn get_current_frame_eval(&self) -> Option<&Vec<(u64, Option<u64>, bool)>> {
        self.frame_eval_data.get(&self.current_frame)
    }

    /// Get ground truth bboxes for current frame: Vec<(gt_track_id, (x,y,w,h))>
    pub fn get_current_frame_gt(&self) -> Option<&Vec<(u64, (f32, f32, f32, f32))>> {
        self.gt_by_display_frame.get(&self.current_frame)
    }

    /// Get Hungarian assignment: tracker_id -> assigned_gt_id
    pub fn get_assignment(&self) -> Option<&HashMap<u64, u64>> {
        self.association_result.as_ref().map(|r| &r.assignment)
    }

    /// Get GT time series: gt_id -> Vec<(frame_idx, Option<pred_id>)>
    pub fn get_gt_timeseries(&self) -> &HashMap<u64, Vec<(usize, Option<u64>)>> {
        &self.gt_timeseries
    }

    /// Get current frame index
    pub fn get_current_frame(&self) -> usize {
        self.current_frame
    }

    /// Get total frame count
    pub fn get_total_frames(&self) -> usize {
        self.frame_count()
    }

    /// Load a human-labeled dataset
    pub fn load_human_labeled(&mut self, json_path: &std::path::Path, timestamps_path: &std::path::Path) {
        // Reset state
        self.clip = None;
        self.decoder = None;
        self.load_error = None;
        self.track_results = None;
        self.current_frame = 0;
        self.video_texture = None;
        self.sampled_frames = Vec::new();
        self.association_result = None;
        self.frame_eval_data = HashMap::new();
        self.gt_by_display_frame = HashMap::new();
        self.gt_timeseries = HashMap::new();

        match Clip::load_human_labeled(json_path, timestamps_path) {
            Ok(mut clip) => {
                // Resolve duplicate GT IDs (same GT ID appearing multiple times in a frame)
                // Keeps only the longest track, nullifies the rest
                let (num_dups, num_nullified) = clip.resolve_duplicate_gt_ids();
                if num_dups > 0 {
                    eprintln!("Resolved {} duplicate GT IDs ({} detections nullified)", num_dups, num_nullified);
                }

                let video_path = clip.video_path.to_string_lossy().to_string();
                eprintln!("Opening video: {}", video_path);

                match VideoMeta::open(&video_path, 960, 540) {
                    Some(meta) => {
                        clip.set_video_dimensions(meta.src_width, meta.src_height);
                        self.video_width = meta.out_width;
                        self.video_height = meta.out_height;

                        let decoder = DecoderHandle::spawn(meta);
                        self.decoder = Some(decoder);
                        self.clip = Some(clip);

                        eprintln!("Loaded human-labeled clip with {} frames (has_ground_truth={})",
                            self.clip.as_ref().unwrap().frame_count,
                            self.clip.as_ref().unwrap().has_ground_truth);

                        // Collect unique classes
                        let mut classes: std::collections::HashSet<String> = std::collections::HashSet::new();
                        for frame_idx in 0..self.clip.as_ref().unwrap().frame_count {
                            for det in self.clip.as_ref().unwrap().get_detections(frame_idx) {
                                classes.insert(det.class.clone());
                            }
                        }
                        let mut class_list: Vec<String> = classes.iter().cloned().collect();
                        class_list.sort();
                        self.overlay_settings.all_classes = class_list;
                        self.overlay_settings.enabled_classes = classes;

                        // Run tracker
                        self.run_tracker();
                    }
                    None => {
                        self.load_error = Some(format!("Failed to open video: {}", video_path));
                    }
                }
            }
            Err(e) => {
                self.load_error = Some(e);
            }
        }
    }

    /// Generate debug log for a specific track at the current frame
    pub fn get_track_debug_log(&self, track_id: usize) -> String {
        let track_results = match &self.track_results {
            Some(r) => r,
            None => return "No track results available".to_string(),
        };

        let debug_info = match track_results.debug_by_frame.get(&self.current_frame) {
            Some(d) => d,
            None => return format!("No debug info for frame {}", self.current_frame),
        };

        let mut log = String::new();
        log.push_str(&format!("=== Track {} Debug Log - Frame {} ===\n\n", track_id, self.current_frame));

        // Find prediction for this track
        log.push_str("--- KALMAN PREDICTION (before matching) ---\n");
        let prediction = debug_info.predictions.iter().find(|p| p.track_id == track_id);
        if let Some(pred) = prediction {
            log.push_str(&format!("  Track ID: {}\n", pred.track_id));
            log.push_str(&format!("  State: {:?}\n", pred.state));
            log.push_str(&format!("  Is Activated: {}\n", pred.is_activated));
            log.push_str(&format!("  Predicted Rect: x={:.1}, y={:.1}, w={:.1}, h={:.1}\n",
                pred.predicted_rect.x(), pred.predicted_rect.y(),
                pred.predicted_rect.width(), pred.predicted_rect.height()));
            log.push_str(&format!("  Velocity: vx={:.4}, vy={:.4}\n", pred.velocity.0, pred.velocity.1));
            log.push_str(&format!("  Cov diag: [x={:.2}, y={:.2}, a={:.6}, h={:.2}]\n",
                pred.covariance_diag[0], pred.covariance_diag[1],
                pred.covariance_diag[2], pred.covariance_diag[3]));
        } else {
            log.push_str(&format!("  Track {} not found in predictions\n", track_id));
        }
        log.push_str("\n");

        // Stage 1: High-conf matching
        log.push_str("--- STAGE 1: High-Conf Matching ---\n");
        let stage1_track_idx = debug_info.stage1.track_pool.iter().position(|t| t.track_id == track_id);
        if let Some(idx) = stage1_track_idx {
            let track_entry = &debug_info.stage1.track_pool[idx];
            log.push_str(&format!("  Pool idx: {}, State: {:?}\n", idx, track_entry.state));
            log.push_str(&format!("  Rect: x={:.1}, y={:.1}, w={:.1}, h={:.1}\n",
                track_entry.rect.x(), track_entry.rect.y(),
                track_entry.rect.width(), track_entry.rect.height()));

            // Check if matched
            let matched = debug_info.stage1.matches.iter().find(|(t, _)| *t == idx);
            if let Some((_, det_idx)) = matched {
                log.push_str(&format!("  MATCHED to detection {}\n", det_idx));
                if *det_idx < debug_info.high_conf_detections.len() {
                    let det = &debug_info.high_conf_detections[*det_idx];
                    log.push_str(&format!("    Det rect: x={:.1}, y={:.1}, w={:.1}, h={:.1}, conf={:.3}\n",
                        det.rect.x(), det.rect.y(), det.rect.width(), det.rect.height(), det.confidence));
                }
            } else if debug_info.stage1.unmatched_tracks.contains(&idx) {
                log.push_str("  UNMATCHED (passed to next stage or lost)\n");
            }

            // Show cost matrix row if available
            if idx < debug_info.stage1.cost_matrix.len() {
                let costs: Vec<String> = debug_info.stage1.cost_matrix[idx].iter()
                    .map(|c| format!("{:.3}", c)).collect();
                log.push_str(&format!("  Cost matrix row: [{}]\n", costs.join(", ")));
            }
        } else {
            log.push_str(&format!("  Track {} not in stage 1 pool\n", track_id));
        }
        log.push_str("\n");

        // Stage 2: Low-conf matching
        log.push_str("--- STAGE 2: Low-Conf Matching ---\n");
        let stage2_track_idx = debug_info.stage2.track_pool.iter().position(|t| t.track_id == track_id);
        if let Some(idx) = stage2_track_idx {
            let track_entry = &debug_info.stage2.track_pool[idx];
            log.push_str(&format!("  Pool idx: {}, State: {:?}\n", idx, track_entry.state));

            let matched = debug_info.stage2.matches.iter().find(|(t, _)| *t == idx);
            if let Some((_, det_idx)) = matched {
                log.push_str(&format!("  MATCHED to low-conf detection {}\n", det_idx));
                if *det_idx < debug_info.low_conf_detections.len() {
                    let det = &debug_info.low_conf_detections[*det_idx];
                    log.push_str(&format!("    Det rect: x={:.1}, y={:.1}, w={:.1}, h={:.1}, conf={:.3}\n",
                        det.rect.x(), det.rect.y(), det.rect.width(), det.rect.height(), det.confidence));
                }
            } else if debug_info.stage2.unmatched_tracks.contains(&idx) {
                log.push_str("  UNMATCHED (will become LOST)\n");
            }

            if idx < debug_info.stage2.cost_matrix.len() {
                let costs: Vec<String> = debug_info.stage2.cost_matrix[idx].iter()
                    .map(|c| format!("{:.3}", c)).collect();
                log.push_str(&format!("  Cost matrix row: [{}]\n", costs.join(", ")));
            }
        } else {
            log.push_str(&format!("  Track {} not in stage 2 pool\n", track_id));
        }
        log.push_str("\n");

        // Stage 3: New track init (for non-activated tracks)
        log.push_str("--- STAGE 3: Track Activation ---\n");
        let stage3_track_idx = debug_info.stage3.track_pool.iter().position(|t| t.track_id == track_id);
        if let Some(idx) = stage3_track_idx {
            let _track_entry = &debug_info.stage3.track_pool[idx];
            log.push_str(&format!("  Pool idx: {} (non-activated track)\n", idx));

            let matched = debug_info.stage3.matches.iter().find(|(t, _)| *t == idx);
            if let Some((_, det_idx)) = matched {
                log.push_str(&format!("  ACTIVATED via detection {}\n", det_idx));
            } else if debug_info.stage3.unmatched_tracks.contains(&idx) {
                log.push_str("  REMOVED (unconfirmed track)\n");
            }
        } else {
            log.push_str(&format!("  Track {} not in stage 3 pool (already activated or not applicable)\n", track_id));
        }
        log.push_str("\n");

        // Final output
        log.push_str("--- FINAL OUTPUT ---\n");
        let output = debug_info.track_outputs.iter().find(|t| t.track_id == track_id);
        if let Some(out) = output {
            log.push_str(&format!("  Track ID: {}\n", out.track_id));
            log.push_str(&format!("  State: {:?}\n", out.state));
            log.push_str(&format!("  Is Activated: {}\n", out.is_activated));
            log.push_str(&format!("  Final Rect: x={:.1}, y={:.1}, w={:.1}, h={:.1}\n",
                out.rect.x(), out.rect.y(), out.rect.width(), out.rect.height()));
            log.push_str(&format!("  Velocity: vx={:.2}, vy={:.2}\n", out.velocity.0, out.velocity.1));
            log.push_str(&format!("  Confidence: {:.3}\n", out.confidence));
        } else {
            log.push_str(&format!("  Track {} not in final output (may be lost/removed)\n", track_id));
        }
        log.push_str("\n");

        // State changes
        log.push_str("--- STATE CHANGES ---\n");
        let changes: Vec<_> = debug_info.state_changes.iter()
            .filter(|(id, _, _)| *id == track_id)
            .collect();
        if changes.is_empty() {
            log.push_str("  No state changes for this track\n");
        } else {
            for (_, from, to) in changes {
                log.push_str(&format!("  {:?} -> {:?}\n", from, to));
            }
        }

        log
    }
}

impl eframe::App for VisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ui::render(self, ctx);
    }
}
