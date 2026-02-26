//! Dataset loader for trio-warehouse format.
//!
//! Expected directory structure:
//! ```
//! sensor_xxx/
//!   video_*.mp4
//!   timestamps_*.json
//!   detections_*.json
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
    #[allow(dead_code)]
    pub timestamp_ms: u64,
    /// Existing track ID from detector (if any)
    #[allow(dead_code)]
    pub track_id: Option<u64>,
    #[allow(dead_code)]
    pub track_vel_x: Option<f32>,
    #[allow(dead_code)]
    pub track_vel_y: Option<f32>,
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

// JSON schema for timestamps file
#[derive(Deserialize)]
struct TimestampsJson {
    #[allow(dead_code)]
    sensor_id: String,
    frame_count: usize,
    timestamps: Vec<u64>,
}

// JSON schema for detections file
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
            };

            detections_by_frame
                .entry(frame_idx)
                .or_insert_with(Vec::new)
                .push(detection);
        }

        Ok(Self {
            video_path,
            frame_count: timestamps_json.frame_count,
            timestamps: timestamps_json.timestamps,
            detections_by_frame,
            video_width: 0,  // Set after video is opened
            video_height: 0,
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
