mod dataset;
mod drawing;
mod ui;
mod video;

use dataset::Clip;
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
        }
    }
}

/// Tracker parameter settings
#[derive(Clone, PartialEq)]
pub struct TrackerParams {
    // Core params
    pub frame_rate: usize,
    pub track_buffer: usize,
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
            track_buffer: 30,
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

            kalman_std_weight_pos: 1.0 / 20.0,
            kalman_std_weight_vel: 1.0 / 160.0,
            kalman_std_weight_position_meas: 1.0 / 20.0,
            kalman_std_weight_position_mot: 1.0 / 20.0,
            kalman_std_weight_velocity_mot: 1.0 / 160.0,
            kalman_std_aspect_ratio_init: 1e-2,
            kalman_std_d_aspect_ratio_init: 1e-5,
            kalman_std_aspect_ratio_mot: 1e-2,
            kalman_std_d_aspect_ratio_mot: 1e-5,
            kalman_std_aspect_ratio_meas: 1e-1,
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
            params.frame_rate,
            params.track_buffer,
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

            // Run tracker update with debug info
            let (tracked, debug_info) = tracker
                .update_with_debug(objects.into_iter())
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

        self.track_results = Some(TrackResults {
            results_by_frame,
            debug_by_frame,
        });
        self.last_tracker_params = self.tracker_params.clone();
        self.last_enabled_classes = self.overlay_settings.enabled_classes.clone();

        // Reset all playback/video state after re-running tracker
        self.current_frame = 0;
        self.mechanics_stage = 0;
        self.video_texture = None;
        self.displayed_original_frame = None;
        self.hovered_track_id = None;
        self.is_playing = false;
        self.last_frame_time = None;

        eprintln!("Tracker finished processing {} sampled frames (from {} total)",
            sampled_indices.len(), clip.frame_count);
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
}

impl eframe::App for VisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ui::render(self, ctx);
    }
}
