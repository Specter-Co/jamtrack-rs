//! Dataset loader supporting multiple formats.
//!
//! Supported formats:
//!
//! 1. Trio-warehouse format (directory-based):
//! ```text
//! sensor_xxx/
//!   video_*.mp4
//!   timestamps_*.json
//!   detections_*.json
//! ```
//!
//! 2. Human-labeled format (single JSON file):
//! ```text
//! {
//!   "video_path": "videos/video_xxx.mp4",
//!   "width": 1920, "height": 1080, "fps": 10.0, "frame_count": 5310,
//!   "entities": [{"id": 0, "name": "person1"}, ...],
//!   "detections": [{"frame_idx": 0, "bbox": [l,t,r,b], "track_id": 1, ...}, ...]
//! }
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// A loaded clip with video, timestamps, and detections
pub struct Clip {
    pub video_path: PathBuf,
    pub frame_count: usize,
    pub timestamps: Vec<u64>,
    pub detections_by_frame: HashMap<usize, Vec<Detection>>,
    pub video_width: u32,
    pub video_height: u32,
    /// Whether this clip has ground truth track IDs
    pub has_ground_truth: bool,
}

/// A single detection
#[derive(Debug, Clone)]
pub struct Detection {
    pub class: String,
    pub confidence: f32,
    /// Normalized ROI (0-1)
    pub roi_top: f32,
    pub roi_left: f32,
    pub roi_bottom: f32,
    pub roi_right: f32,
    pub timestamp_ms: u64,
    /// Existing track ID from detector (if any)
    pub track_id: Option<u64>,
    pub track_vel_x: Option<f32>,
    pub track_vel_y: Option<f32>,
    /// Ground truth track ID (for evaluation)
    pub gt_track_id: Option<u64>,
}

impl Detection {
    /// Convert normalized ROI to pixel coordinates
    pub fn to_pixel_rect(&self, width: u32, height: u32) -> (f32, f32, f32, f32) {
        let x = self.roi_left * width as f32;
        let y = self.roi_top * height as f32;
        let w = (self.roi_right - self.roi_left) * width as f32;
        let h = (self.roi_bottom - self.roi_top) * height as f32;
        (x, y, w, h)
    }
}

// JSON schema for timestamps file (trio-warehouse format)
#[derive(Deserialize)]
struct TimestampsJson {
    #[allow(dead_code)]
    sensor_id: String,
    frame_count: usize,
    timestamps: Vec<u64>,
}

// JSON schema for detections file (trio-warehouse format)
#[derive(Deserialize)]
struct DetectionsJson {
    #[allow(dead_code)]
    metadata: DetectionsMetadata,
    detections: Vec<DetectionJson>,
}

#[derive(Deserialize)]
struct DetectionsMetadata {
    #[allow(dead_code)]
    sensor_id: String,
}

#[derive(Deserialize)]
struct DetectionJson {
    class: String,
    confidence: f32,
    roi_top: f32,
    roi_left: f32,
    roi_bottom: f32,
    roi_right: f32,
    timestamp_ms: u64,
    track_id: Option<u64>,
    track_vel_x: Option<f32>,
    track_vel_y: Option<f32>,
}

// JSON schema for human-labeled format
#[derive(Deserialize)]
struct HumanLabeledJson {
    video_path: String,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    #[allow(dead_code)]
    fps: f32,
    #[allow(dead_code)]
    prompt: Option<String>,
    detections: Vec<HumanLabeledDetectionJson>,
}

#[derive(Deserialize)]
struct HumanLabeledDetectionJson {
    frame_idx: usize,
    #[serde(alias = "timestamp_ms")]
    timestamp: u64,
    /// [left, top, right, bottom] normalized (0-1)
    bbox: [f32; 4],
    confidence: f32,
    label: String,
    track_id: Option<u64>,
    /// Ground truth entity ID (this is the true identity)
    entity_id: Option<u64>,
    #[serde(default)]
    deleted: bool,
}

// JSON schema for timestamps file (used by human-labeled format)
#[derive(Deserialize)]
struct TimestampsJsonMinimal {
    frame_count: usize,
    timestamps: Vec<u64>,
}

impl Clip {
    /// Load a clip from a directory containing video_*.mp4, timestamps_*.json, detections_*.json
    pub fn load(dir: &Path) -> Result<Self, String> {
        // Find files by pattern
        let video_path = find_file_with_prefix(dir, "video_", ".mp4")?;
        let timestamps_path = find_file_with_prefix(dir, "timestamps_", ".json")?;
        let detections_path = find_file_with_prefix(dir, "detections_", ".json")?;

        // Load timestamps
        let timestamps_content = fs::read_to_string(&timestamps_path)
            .map_err(|e| format!("Failed to read timestamps: {}", e))?;
        let timestamps_json: TimestampsJson = serde_json::from_str(&timestamps_content)
            .map_err(|e| format!("Failed to parse timestamps: {}", e))?;

        // Load detections
        let detections_content = fs::read_to_string(&detections_path)
            .map_err(|e| format!("Failed to read detections: {}", e))?;
        let detections_json: DetectionsJson = serde_json::from_str(&detections_content)
            .map_err(|e| format!("Failed to parse detections: {}", e))?;

        // Build timestamp -> frame index mapping
        let timestamp_to_frame: HashMap<u64, usize> = timestamps_json
            .timestamps
            .iter()
            .enumerate()
            .map(|(i, &ts)| (ts, i))
            .collect();

        // Group detections by frame
        let mut detections_by_frame: HashMap<usize, Vec<Detection>> = HashMap::new();
        for det_json in detections_json.detections {
            let frame_idx = timestamp_to_frame
                .get(&det_json.timestamp_ms)
                .copied()
                .unwrap_or_else(|| {
                    // Find nearest frame if exact match not found
                    find_nearest_frame(&timestamps_json.timestamps, det_json.timestamp_ms)
                });

            let detection = Detection {
                class: det_json.class,
                confidence: det_json.confidence,
                roi_top: det_json.roi_top,
                roi_left: det_json.roi_left,
                roi_bottom: det_json.roi_bottom,
                roi_right: det_json.roi_right,
                timestamp_ms: det_json.timestamp_ms,
                track_id: det_json.track_id,
                track_vel_x: det_json.track_vel_x,
                track_vel_y: det_json.track_vel_y,
                gt_track_id: None,
            };

            detections_by_frame
                .entry(frame_idx)
                .or_default()
                .push(detection);
        }

        Ok(Self {
            video_path,
            frame_count: timestamps_json.frame_count,
            timestamps: timestamps_json.timestamps,
            detections_by_frame,
            video_width: 0,  // Set after video is opened
            video_height: 0,
            has_ground_truth: false,
        })
    }

    /// Load a clip from a human-labeled JSON file.
    /// The JSON file should contain video_path, detections, and optionally entities.
    /// Timestamps are loaded from a separate timestamps JSON file.
    pub fn load_human_labeled(
        json_path: &Path,
        timestamps_path: &Path,
    ) -> Result<Self, String> {
        // Load human-labeled JSON
        let json_content = fs::read_to_string(json_path)
            .map_err(|e| format!("Failed to read human-labeled JSON: {}", e))?;
        let labeled: HumanLabeledJson = serde_json::from_str(&json_content)
            .map_err(|e| format!("Failed to parse human-labeled JSON: {}", e))?;

        // Load timestamps
        let timestamps_content = fs::read_to_string(timestamps_path)
            .map_err(|e| format!("Failed to read timestamps: {}", e))?;
        let timestamps_json: TimestampsJsonMinimal = serde_json::from_str(&timestamps_content)
            .map_err(|e| format!("Failed to parse timestamps: {}", e))?;

        // Resolve video path relative to JSON file's directory
        let json_dir = json_path.parent().unwrap_or(Path::new("."));
        let video_path = json_dir.join(&labeled.video_path);

        // Group detections by frame, filtering out deleted ones
        let mut detections_by_frame: HashMap<usize, Vec<Detection>> = HashMap::new();
        let mut warned_missing_entity_id = false;
        for det in labeled.detections {
            if det.deleted {
                continue;
            }

            // entity_id is the ground truth track ID; fall back to track_id with warning
            let gt_track_id = match det.entity_id {
                Some(id) => Some(id),
                None => {
                    if !warned_missing_entity_id && det.track_id.is_some() {
                        eprintln!("Warning: entity_id not found, falling back to track_id for ground truth");
                        warned_missing_entity_id = true;
                    }
                    det.track_id
                }
            };

            let detection = Detection {
                class: det.label,
                confidence: det.confidence,
                // bbox is [left, top, right, bottom]
                roi_left: det.bbox[0],
                roi_top: det.bbox[1],
                roi_right: det.bbox[2],
                roi_bottom: det.bbox[3],
                timestamp_ms: det.timestamp,
                track_id: det.track_id,
                track_vel_x: None,
                track_vel_y: None,
                gt_track_id,
            };

            detections_by_frame
                .entry(det.frame_idx)
                .or_default()
                .push(detection);
        }

        Ok(Self {
            video_path,
            frame_count: timestamps_json.frame_count,
            timestamps: timestamps_json.timestamps,
            detections_by_frame,
            video_width: 0,
            video_height: 0,
            has_ground_truth: true,
        })
    }

    /// Get detections for a specific frame
    pub fn get_detections(&self, frame_idx: usize) -> &[Detection] {
        self.detections_by_frame
            .get(&frame_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get timestamp for a frame
    pub fn get_timestamp(&self, frame_idx: usize) -> Option<u64> {
        self.timestamps.get(frame_idx).copied()
    }

    /// Update video dimensions after opening
    pub fn set_video_dimensions(&mut self, width: u32, height: u32) {
        self.video_width = width;
        self.video_height = height;
    }

    /// Returns frame indices to process at the target FPS.
    /// If target_fps is 0 or >= native fps, returns all frames.
    pub fn get_sampled_frame_indices(&self, target_fps: f32) -> Vec<usize> {
        if target_fps <= 0.0 || self.frame_count < 2 {
            return (0..self.frame_count).collect();
        }

        // Calculate native FPS from timestamps
        let first_ts = self.timestamps.first().copied().unwrap_or(0);
        let last_ts = self.timestamps.last().copied().unwrap_or(0);
        let duration_ms = last_ts.saturating_sub(first_ts);

        if duration_ms == 0 {
            return (0..self.frame_count).collect();
        }

        let native_fps = (self.frame_count as f64 * 1000.0) / duration_ms as f64;

        if target_fps as f64 >= native_fps {
            return (0..self.frame_count).collect();
        }

        // Sample frames at target FPS intervals
        let step = native_fps / target_fps as f64;
        let mut indices = Vec::new();
        let mut pos = 0.0;
        while (pos as usize) < self.frame_count {
            indices.push(pos as usize);
            pos += step;
        }
        indices
    }
}

fn find_file_with_prefix(dir: &Path, prefix: &str, suffix: &str) -> Result<PathBuf, String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(prefix) && name_str.ends_with(suffix) {
            return Ok(entry.path());
        }
    }

    Err(format!("No file matching {}*{} found in {:?}", prefix, suffix, dir))
}

fn find_nearest_frame(timestamps: &[u64], target_ms: u64) -> usize {
    let mut best_idx = 0;
    let mut best_diff = u64::MAX;

    for (i, &ts) in timestamps.iter().enumerate() {
        let diff = if ts > target_ms { ts - target_ms } else { target_ms - ts };
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }

    best_idx
}

/// Compute IoU between two bboxes (roi format: left, top, right, bottom)
fn compute_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (a_left, a_top, a_right, a_bottom) = a;
    let (b_left, b_top, b_right, b_bottom) = b;

    let inter_left = a_left.max(b_left);
    let inter_top = a_top.max(b_top);
    let inter_right = a_right.min(b_right);
    let inter_bottom = a_bottom.min(b_bottom);

    if inter_right <= inter_left || inter_bottom <= inter_top {
        return 0.0;
    }

    let inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
    let a_area = (a_right - a_left) * (a_bottom - a_top);
    let b_area = (b_right - b_left) * (b_bottom - b_top);
    let union_area = a_area + b_area - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

impl Clip {
    /// Resolve duplicate GT IDs within the same frame.
    ///
    /// When a GT ID appears multiple times in the same frame, this function:
    /// 1. Splits them into separate pseudo-tracks using IoU-based tracking
    /// 2. Keeps only the longest pseudo-track with the original GT ID
    /// 3. Sets gt_track_id to None for all other (shorter) pseudo-tracks
    ///
    /// Returns the number of GT IDs that had duplicates, and the number of detections nullified.
    pub fn resolve_duplicate_gt_ids(&mut self) -> (usize, usize) {
        // Step 1: Find all GT IDs that appear more than once in any frame
        let mut duplicate_gt_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut all_gt_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for dets in self.detections_by_frame.values() {
            let mut frame_gt_counts: HashMap<u64, usize> = HashMap::new();
            for det in dets {
                if let Some(gt_id) = det.gt_track_id {
                    all_gt_ids.insert(gt_id);
                    *frame_gt_counts.entry(gt_id).or_insert(0) += 1;
                }
            }
            for (gt_id, count) in frame_gt_counts {
                if count > 1 {
                    duplicate_gt_ids.insert(gt_id);
                }
            }
        }

        if duplicate_gt_ids.is_empty() {
            return (0, 0);
        }

        eprintln!("[WARN] Found {} GT IDs with duplicates in same frame: {:?}",
            duplicate_gt_ids.len(),
            duplicate_gt_ids.iter().collect::<Vec<_>>());

        // Step 2: Find the next available ID (max existing + 1) for temporary pseudo-IDs
        let max_existing_id = all_gt_ids.iter().copied().max().unwrap_or(0);
        let mut next_pseudo_id = max_existing_id + 1;

        // Step 3: For each duplicate GT ID, track instances across frames
        let mut frame_indices: Vec<usize> = self.detections_by_frame.keys().copied().collect();
        frame_indices.sort();

        let mut total_nullified = 0;

        for &original_gt_id in &duplicate_gt_ids {
            // Track active pseudo-IDs and their last known positions
            // pseudo_id -> (last_frame, bbox)
            let mut active_tracks: HashMap<u64, (usize, (f32, f32, f32, f32))> = HashMap::new();

            // Track which (frame_idx, det_idx) belong to which pseudo_id
            let mut pseudo_id_detections: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();

            // Process frames in order to build pseudo-tracks
            for &frame_idx in &frame_indices {
                let dets = match self.detections_by_frame.get(&frame_idx) {
                    Some(d) => d,
                    None => continue,
                };

                // Find all detections with this GT ID in this frame
                let mut det_indices: Vec<usize> = Vec::new();
                for (i, det) in dets.iter().enumerate() {
                    if det.gt_track_id == Some(original_gt_id) {
                        det_indices.push(i);
                    }
                }

                if det_indices.is_empty() {
                    continue;
                }

                // Get bboxes for these detections
                let det_bboxes: Vec<(f32, f32, f32, f32)> = det_indices.iter()
                    .map(|&i| {
                        let d = &dets[i];
                        (d.roi_left, d.roi_top, d.roi_right, d.roi_bottom)
                    })
                    .collect();

                // If only one detection and no active tracks, assign original ID
                if det_indices.len() == 1 && active_tracks.is_empty() {
                    active_tracks.insert(original_gt_id, (frame_idx, det_bboxes[0]));
                    pseudo_id_detections.entry(original_gt_id).or_default().push((frame_idx, det_indices[0]));
                    continue;
                }

                // Match detections to active tracks using IoU (greedy)
                let mut assigned_pseudo_ids: Vec<Option<u64>> = vec![None; det_indices.len()];
                let mut used_tracks: std::collections::HashSet<u64> = std::collections::HashSet::new();

                // Build IoU pairs and sort by IoU descending
                let mut iou_pairs: Vec<(f32, usize, u64)> = Vec::new();
                for (det_i, &bbox) in det_bboxes.iter().enumerate() {
                    for (&pseudo_id, &(_, track_bbox)) in &active_tracks {
                        let iou = compute_iou(bbox, track_bbox);
                        if iou > 0.1 {
                            iou_pairs.push((iou, det_i, pseudo_id));
                        }
                    }
                }
                iou_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Greedy assignment
                for (_, det_i, pseudo_id) in iou_pairs {
                    if assigned_pseudo_ids[det_i].is_none() && !used_tracks.contains(&pseudo_id) {
                        assigned_pseudo_ids[det_i] = Some(pseudo_id);
                        used_tracks.insert(pseudo_id);
                    }
                }

                // Assign new pseudo-IDs to unmatched detections
                for assigned in assigned_pseudo_ids.iter_mut() {
                    if assigned.is_none() {
                        if !used_tracks.contains(&original_gt_id) && !active_tracks.contains_key(&original_gt_id) {
                            *assigned = Some(original_gt_id);
                            used_tracks.insert(original_gt_id);
                        } else {
                            *assigned = Some(next_pseudo_id);
                            next_pseudo_id += 1;
                        }
                    }
                }

                // Record assignments and update active tracks
                for (det_i, &det_idx) in det_indices.iter().enumerate() {
                    let pseudo_id = assigned_pseudo_ids[det_i].unwrap();
                    pseudo_id_detections.entry(pseudo_id).or_default().push((frame_idx, det_idx));
                    active_tracks.insert(pseudo_id, (frame_idx, det_bboxes[det_i]));
                }
            }

            // Step 4: Find the longest pseudo-track and keep only that one
            let longest_pseudo_id = pseudo_id_detections.iter()
                .max_by_key(|(_, dets)| dets.len())
                .map(|(&id, _)| id);

            // Nullify all detections that don't belong to the longest track
            for (&pseudo_id, det_locations) in &pseudo_id_detections {
                if Some(pseudo_id) != longest_pseudo_id {
                    let frames_pruned: Vec<usize> = det_locations.iter().map(|(f, _)| *f).collect();
                    eprintln!("[WARN] GT ID {}: pruning {} detections on frames {:?} (keeping pseudo-track {:?} with {} detections)",
                        original_gt_id,
                        det_locations.len(),
                        frames_pruned,
                        longest_pseudo_id,
                        longest_pseudo_id.map(|id| pseudo_id_detections.get(&id).map(|d| d.len()).unwrap_or(0)).unwrap_or(0));
                    for &(frame_idx, det_idx) in det_locations {
                        if let Some(dets) = self.detections_by_frame.get_mut(&frame_idx) {
                            if let Some(det) = dets.get_mut(det_idx) {
                                det.gt_track_id = None;
                                total_nullified += 1;
                            }
                        }
                    }
                }
            }
        }

        (duplicate_gt_ids.len(), total_nullified)
    }
}
