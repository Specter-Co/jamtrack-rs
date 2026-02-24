//! Debug information structures for visualizing ByteTracker internals.
//!
//! These structures capture intermediate state during tracking for debugging
//! and visualization purposes.

use crate::rect::Rect;

/// Track state for visualization (mirrors internal STrackState)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

/// Complete debug information for a single frame's tracking update
#[derive(Debug, Clone)]
pub struct FrameDebugInfo {
    pub frame_id: usize,

    /// Input detections split by confidence tier
    pub high_conf_detections: Vec<DetectionInfo>,
    pub low_conf_detections: Vec<DetectionInfo>,

    /// Kalman predictions (before any matching)
    pub predictions: Vec<PredictionInfo>,

    /// Stage 1: High-conf matching to active+lost tracks
    pub stage1: AssociationStageInfo,

    /// Stage 2: Low-conf matching to remaining active tracks
    pub stage2: AssociationStageInfo,

    /// Stage 3: New track initialization from remaining detections
    pub stage3: AssociationStageInfo,

    /// Final track outputs after all stages
    pub track_outputs: Vec<TrackOutputInfo>,

    /// State transitions that occurred: (track_id, from_state, to_state)
    pub state_changes: Vec<(usize, TrackState, TrackState)>,
}

impl Default for FrameDebugInfo {
    fn default() -> Self {
        Self {
            frame_id: 0,
            high_conf_detections: Vec::new(),
            low_conf_detections: Vec::new(),
            predictions: Vec::new(),
            stage1: AssociationStageInfo::default(),
            stage2: AssociationStageInfo::default(),
            stage3: AssociationStageInfo::default(),
            track_outputs: Vec::new(),
            state_changes: Vec::new(),
        }
    }
}

/// Information about a single detection
#[derive(Debug, Clone)]
pub struct DetectionInfo {
    pub detection_id: i64,
    pub rect: Rect<f32>,
    pub confidence: f32,
}

/// Information about a track's Kalman prediction
#[derive(Debug, Clone)]
pub struct PredictionInfo {
    pub track_id: usize,
    pub predicted_rect: Rect<f32>,
    pub velocity: (f32, f32),
    /// Diagonal of position covariance (x, y, aspect_ratio, height uncertainty)
    pub covariance_diag: [f32; 4],
    pub state: TrackState,
    pub is_activated: bool,
}

/// Information about one association stage
#[derive(Debug, Clone, Default)]
pub struct AssociationStageInfo {
    /// Track indices that participated in this stage
    pub track_pool: Vec<TrackPoolEntry>,
    /// Detection indices that participated in this stage
    pub detection_pool: Vec<usize>,
    /// Cost matrix (1 - IoU * weight - conf * (1-weight))
    /// Rows = tracks, Cols = detections
    pub cost_matrix: Vec<Vec<f32>>,
    /// Successful matches: (track_pool_idx, detection_pool_idx)
    pub matches: Vec<(usize, usize)>,
    /// Track indices that didn't match
    pub unmatched_tracks: Vec<usize>,
    /// Detection indices that didn't match
    pub unmatched_detections: Vec<usize>,
}

/// Entry in a track pool for association
#[derive(Debug, Clone)]
pub struct TrackPoolEntry {
    pub track_id: usize,
    pub rect: Rect<f32>,
    pub state: TrackState,
}

/// Information about a track in the final output
#[derive(Debug, Clone)]
pub struct TrackOutputInfo {
    pub track_id: usize,
    pub rect: Rect<f32>,
    pub velocity: (f32, f32),
    pub confidence: f32,
    pub state: TrackState,
    pub is_activated: bool,
}
