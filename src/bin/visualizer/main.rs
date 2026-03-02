mod drawing;
mod ui;
mod video;

use jamtrack_rs::dataset::Clip;
use jamtrack_rs::evaluation::{Evaluator, EvalMetrics, GtDetection, PredDetection, FrameEvalResult};
use eframe::egui;
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::debug_info::FrameDebugInfo;
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;
use std::collections::HashMap;
use std::path::PathBuf;
use video::{DecoderHandle, VideoMeta};

fn main() -> eframe::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let clip_dir = args.get(1).map(PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("ByteTrack Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "ByteTrack Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(VisualizerApp::new(cc, clip_dir)))),
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
    pub show_id_switches: bool,
    pub show_false_positives: bool,
    pub show_false_negatives: bool,
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
            show_id_switches: true,
            show_false_positives: true,
            show_false_negatives: true,
            show_gt_boxes: true,
        }
    }
}

/// Tracker parameter settings
#[derive(Clone, PartialEq)]
pub struct TrackerParams {
    // Core params
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
    evaluator: Option<Evaluator>,
    eval_metrics: Option<EvalMetrics>,
    // Ground truth bboxes keyed by display frame index
    gt_by_display_frame: HashMap<usize, Vec<(u64, (f32, f32, f32, f32))>>,
}

impl VisualizerApp {
    fn new(_cc: &eframe::CreationContext<'_>, clip_dir: Option<PathBuf>) -> Self {
        let tracker_params = TrackerParams::default();
        let mut app = Self {
            clip: None,
            decoder: None,
            load_error: None,
            track_results: None,
            last_tracker_params: tracker_params.clone(),
            last_enabled_classes: std::collections::HashSet::new(),
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
            evaluator: None,
            eval_metrics: None,
            gt_by_display_frame: HashMap::new(),
        };

        if let Some(dir) = clip_dir {
            app.load_clip(&dir);
        }

        app
    }

    fn load_clip(&mut self, dir: &std::path::Path) {
        // Reset state
        self.clip = None;
        self.decoder = None;
        self.load_error = None;
        self.track_results = None;
        self.current_frame = 0;
        self.video_texture = None;
        self.sampled_frames = Vec::new();
        self.evaluator = None;
        self.eval_metrics = None;
        self.gt_by_display_frame = HashMap::new();

        // Load clip metadata
        match Clip::load(dir) {
            Ok(mut clip) => {
                // Open video decoder
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

                        eprintln!("Loaded clip with {} frames", self.clip.as_ref().unwrap().frame_count);

                        // Collect unique classes from detections
                        let mut classes: std::collections::HashSet<String> = std::collections::HashSet::new();
                        for frame_idx in 0..self.clip.as_ref().unwrap().frame_count {
                            for det in self.clip.as_ref().unwrap().get_detections(frame_idx) {
                                classes.insert(det.class.clone());
                            }
                        }
                        let mut class_list: Vec<String> = classes.iter().cloned().collect();
                        class_list.sort();
                        self.overlay_settings.all_classes = class_list.clone();
                        self.overlay_settings.enabled_classes = classes; // Enable all by default

                        // Run ByteTrack on detections
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
            // Fallback uses 100ms intervals (default dt) when timestamp unavailable
            let timestamp_ms: u64 = clip.get_timestamp(frame_idx)
                .unwrap_or(display_idx as u64 * 100);

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
                self.evaluator = None;
                self.eval_metrics = None;
                self.gt_by_display_frame = HashMap::new();
                return;
            }
        };

        let mut evaluator = Evaluator::new(0.5);  // IoU threshold for matching
        let mut gt_by_display_frame: HashMap<usize, Vec<(u64, (f32, f32, f32, f32))>> = HashMap::new();

        // Collect ground truth and predictions
        for (display_idx, &frame_idx) in sampled_indices.iter().enumerate() {
            let detections = clip.get_detections(frame_idx);

            // Ground truth: detections with gt_track_id
            let mut gt_dets = Vec::new();
            let mut gt_bboxes = Vec::new();
            for det in detections {
                if let Some(gt_id) = det.gt_track_id {
                    let (x, y, w, h) = det.to_pixel_rect(clip.video_width, clip.video_height);
                    gt_dets.push(GtDetection {
                        frame_idx: display_idx,
                        track_id: gt_id,
                        bbox: (x, y, w, h),
                    });
                    gt_bboxes.push((gt_id, (x, y, w, h)));
                }
            }
            gt_by_display_frame.insert(display_idx, gt_bboxes);
            evaluator.add_ground_truth(gt_dets);

            // Predictions: tracked objects
            if let Some(tracks) = results_by_frame.get(&display_idx) {
                let pred_dets: Vec<PredDetection> = tracks.iter().map(|t| {
                    PredDetection {
                        frame_idx: display_idx,
                        track_id: t.track_id as u64,
                        bbox: (t.rect.x(), t.rect.y(), t.rect.width(), t.rect.height()),
                    }
                }).collect();
                evaluator.add_predictions(pred_dets);
            }
        }

        // Evaluate each frame
        for display_idx in 0..sampled_indices.len() {
            evaluator.evaluate_frame(display_idx);
        }

        // Compute HOTA metrics
        evaluator.compute_hota();
        let metrics = evaluator.metrics().clone();

        eprintln!("Evaluation complete:");
        eprintln!("  MOTA: {:.2}%", metrics.mota() * 100.0);
        eprintln!("  HOTA: {:.2}%", metrics.hota() * 100.0);
        eprintln!("  IDF1: {:.2}%", metrics.idf1() * 100.0);
        eprintln!("  Precision: {:.2}%", metrics.precision() * 100.0);
        eprintln!("  Recall: {:.2}%", metrics.recall() * 100.0);
        eprintln!("  ID Switches: {}", metrics.id_switches);
        eprintln!("  FP: {}, FN: {}, TP: {}", metrics.false_positives, metrics.false_negatives, metrics.true_positives);

        self.evaluator = Some(evaluator);
        self.eval_metrics = Some(metrics);
        self.gt_by_display_frame = gt_by_display_frame;
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

    /// Get evaluation metrics
    pub fn get_eval_metrics(&self) -> Option<&EvalMetrics> {
        self.eval_metrics.as_ref()
    }

    /// Get frame evaluation result for the current frame
    pub fn get_current_frame_eval(&self) -> Option<&FrameEvalResult> {
        self.evaluator.as_ref()?.frame_result(self.current_frame)
    }

    /// Get ground truth bboxes for current frame: Vec<(gt_track_id, (x,y,w,h))>
    pub fn get_current_frame_gt(&self) -> Option<&Vec<(u64, (f32, f32, f32, f32))>> {
        self.gt_by_display_frame.get(&self.current_frame)
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
        self.evaluator = None;
        self.eval_metrics = None;
        self.gt_by_display_frame = HashMap::new();

        match Clip::load_human_labeled(json_path, timestamps_path) {
            Ok(mut clip) => {
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
            log.push_str(&format!("  Pos cov diag: [x={:.2}, y={:.2}, a={:.6}, h={:.2}]\n",
                pred.covariance_diag[0], pred.covariance_diag[1],
                pred.covariance_diag[2], pred.covariance_diag[3]));
            log.push_str(&format!("  Vel cov diag: [vx={:.2}, vy={:.2}, va={:.6}, vh={:.2}]\n",
                pred.covariance_diag[4], pred.covariance_diag[5],
                pred.covariance_diag[6], pred.covariance_diag[7]));
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
