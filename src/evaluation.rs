//! Association Score evaluation for multi-object tracking.
//!
//! This metric evaluates tracking quality using a many-to-1 matching from
//! predicted track IDs to ground truth IDs based on frequency.

use std::collections::HashMap;

/// A frame-level association: tracker output matched to a detection
#[derive(Debug, Clone)]
pub struct FrameAssociation {
    pub frame_idx: usize,
    pub tracker_id: u64,
    /// GT track ID if the detection has one, None otherwise
    pub gt_track_id: Option<u64>,
}

/// Association Score metrics
#[derive(Debug, Clone, Default)]
pub struct AssociationMetrics {
    /// Matched: (gt_id, pred_id) where pred_id is assigned to this gt_id
    pub matched: usize,
    /// False match: (gt_id, pred_id) where pred_id is assigned to a different gt_id
    pub false_match: usize,
    /// Untracked: (gt_id, pred_id=NULL) - GT object not tracked
    pub untracked: usize,
    /// Tracked with no object: count of unique pred_ids not assigned to any GT
    pub tracked_no_object: usize,
    /// Total pairs processed (excluding NULL, NULL)
    pub total_pairs: usize,
    /// Number of unique GT tracks
    pub num_gt_tracks: usize,
    /// Number of unique tracker IDs
    pub num_tracker_ids: usize,
    /// Fragmentation: RMS of total ID switches per GT (any pred_id change)
    pub fragmentation: f32,
    /// Confusion: RMS of bad ID switches per GT (switch TO pred_id not assigned to this GT)
    pub confusion: f32,
    /// Total switches across all GTs (sum, for normalization)
    pub total_switches: usize,
    /// Total bad switches across all GTs (sum, for normalization)
    pub total_bad_switches: usize,
}

impl AssociationMetrics {
    /// Coverage = Matched / (Matched + FalseMatch + Untracked)
    pub fn coverage(&self) -> f32 {
        let denom = self.matched + self.false_match + self.untracked;
        if denom == 0 { 0.0 } else { self.matched as f32 / denom as f32 }
    }
}

/// Compute RMS from a slice of counts (used for aggregating across sequences)
pub fn compute_rms(counts: &[usize]) -> f32 {
    if counts.is_empty() {
        0.0
    } else {
        let sum_sq: usize = counts.iter().map(|&c| c * c).sum();
        let mean_sq = sum_sq as f32 / counts.len() as f32;
        mean_sq.sqrt()
    }
}

/// Association Score result including the assignment mapping
#[derive(Debug, Clone, Default)]
pub struct AssociationResult {
    pub metrics: AssociationMetrics,
    /// Many-to-1 assignment: pred_id -> assigned gt_id (by frequency)
    pub assignment: HashMap<u64, u64>,
    /// Per-GT total switch counts (for aggregation across sequences)
    pub total_switches_per_gt: Vec<usize>,
    /// Per-GT bad switch counts (for aggregation across sequences)
    pub bad_switches_per_gt: Vec<usize>,
}

/// Evaluator for Association Score tracking metric
pub struct Evaluator {
    /// All frame-level associations
    associations: Vec<FrameAssociation>,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            associations: Vec::new(),
        }
    }

    /// Add a frame-level association (tracker output matched to detection)
    pub fn add_association(&mut self, assoc: FrameAssociation) {
        self.associations.push(assoc);
    }

    /// Compute Association Score using frequency-based many-to-1 matching
    pub fn compute_association_score(&self) -> AssociationResult {
        // Count (pred_id, gt_id) co-occurrences for matching
        let mut pred_gt_counts: HashMap<u64, HashMap<u64, usize>> = HashMap::new();
        // pred_id -> (gt_id -> count)

        let mut all_gt_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut all_pred_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for assoc in &self.associations {
            all_pred_ids.insert(assoc.tracker_id);
            if let Some(gt_id) = assoc.gt_track_id {
                all_gt_ids.insert(gt_id);
                *pred_gt_counts
                    .entry(assoc.tracker_id)
                    .or_default()
                    .entry(gt_id)
                    .or_default() += 1;
            }
        }

        // Step 1: Many-to-1 matching - for each pred_id, find most frequent gt_id
        let mut assignment: HashMap<u64, u64> = HashMap::new();
        for (&pred_id, gt_counts) in &pred_gt_counts {
            if let Some((&best_gt, _)) = gt_counts.iter().max_by_key(|(_, &count)| count) {
                assignment.insert(pred_id, best_gt);
            }
        }

        // Step 2: Categorize all pairs
        let mut matched = 0usize;
        let mut false_match = 0usize;
        let untracked = 0usize;  // Computed separately via compute_association_score_with_untracked
        let mut total_pairs = 0usize;

        for assoc in &self.associations {
            let gt_id = assoc.gt_track_id;
            let pred_id = assoc.tracker_id;

            // Skip (NULL, NULL) - but we don't have that case since pred_id is always present
            // In our model, pred_id always exists (tracker output), gt_id may be None

            total_pairs += 1;

            match gt_id {
                Some(gt) => {
                    // GT exists
                    match assignment.get(&pred_id) {
                        Some(&assigned_gt) if assigned_gt == gt => {
                            // Matched: pred_id is assigned to this gt_id
                            matched += 1;
                        }
                        _ => {
                            // FalseMatch: pred_id is assigned to a different gt_id (or not assigned)
                            false_match += 1;
                        }
                    }
                }
                None => {
                    // GT is NULL - tracker tracking something with no GT
                    // This counts as false_match if pred_id is assigned to some GT
                    // Otherwise it's just noise (pred_id not assigned to anything)
                    if assignment.contains_key(&pred_id) {
                        false_match += 1;
                    }
                    // If not assigned, we don't count it in matched/false_match
                }
            }
        }

        // tracked_no_object = count of unique pred_ids not assigned to any GT
        let tracked_no_object = all_pred_ids.len() - assignment.len();

        // We also need to count UNTRACKED - GT objects that weren't tracked
        // This requires knowing when a GT object appeared but had no pred_id
        // But our associations only include cases where tracker output exists...
        // We need a different data structure to capture "GT present but no tracker"

        // For now, we can only compute untracked if we have separate GT-only data
        // The current FrameAssociation model assumes tracker output always exists
        // Let's add a way to track untracked GTs

        // Step 3: Compute fragmentation and confusion
        // For each GT, collect (frame_idx, pred_id) pairs, sort by frame
        // Fragmentation: count ALL pred_id changes
        // Confusion: count switches TO a pred_id NOT assigned to this GT
        let mut gt_to_frame_preds: HashMap<u64, Vec<(usize, u64)>> = HashMap::new();
        for assoc in &self.associations {
            if let Some(gt_id) = assoc.gt_track_id {
                gt_to_frame_preds
                    .entry(gt_id)
                    .or_default()
                    .push((assoc.frame_idx, assoc.tracker_id));
            }
        }

        let mut total_switch_counts: Vec<usize> = Vec::new();
        let mut bad_switch_counts: Vec<usize> = Vec::new();
        for (gt_id, mut frame_preds) in gt_to_frame_preds {
            // Sort by frame index
            frame_preds.sort_by_key(|(frame, _)| *frame);

            let mut total_switches = 0usize;
            let mut bad_switches = 0usize;
            for i in 1..frame_preds.len() {
                let new_pred_id = frame_preds[i].1;
                let old_pred_id = frame_preds[i - 1].1;
                if new_pred_id != old_pred_id {
                    total_switches += 1;
                    // Check if new_pred_id is NOT assigned to this gt_id
                    let new_assigned_to = assignment.get(&new_pred_id);
                    if new_assigned_to != Some(&gt_id) {
                        bad_switches += 1;
                    }
                }
            }
            total_switch_counts.push(total_switches);
            bad_switch_counts.push(bad_switches);
        }

        let fragmentation = if total_switch_counts.is_empty() {
            0.0
        } else {
            let sum_sq: usize = total_switch_counts.iter().map(|&c| c * c).sum();
            let mean_sq = sum_sq as f32 / total_switch_counts.len() as f32;
            mean_sq.sqrt()
        };

        let confusion = if bad_switch_counts.is_empty() {
            0.0
        } else {
            let sum_sq: usize = bad_switch_counts.iter().map(|&c| c * c).sum();
            let mean_sq = sum_sq as f32 / bad_switch_counts.len() as f32;
            mean_sq.sqrt()
        };

        let total_switches: usize = total_switch_counts.iter().sum();
        let total_bad_switches: usize = bad_switch_counts.iter().sum();

        AssociationResult {
            metrics: AssociationMetrics {
                matched,
                false_match,
                untracked,
                tracked_no_object,
                total_pairs,
                num_gt_tracks: all_gt_ids.len(),
                num_tracker_ids: all_pred_ids.len(),
                fragmentation,
                confusion,
                total_switches,
                total_bad_switches,
            },
            assignment,
            total_switches_per_gt: total_switch_counts,
            bad_switches_per_gt: bad_switch_counts,
        }
    }

    /// Add untracked GT count (GT objects that had no tracker output)
    /// Call this after processing all associations to account for missed GTs
    pub fn compute_association_score_with_untracked(&self, gt_frame_counts: &HashMap<u64, usize>) -> AssociationResult {
        let mut result = self.compute_association_score();

        // Count how many times each GT was tracked
        let mut gt_tracked_counts: HashMap<u64, usize> = HashMap::new();
        for assoc in &self.associations {
            if let Some(gt_id) = assoc.gt_track_id {
                *gt_tracked_counts.entry(gt_id).or_default() += 1;
            }
        }

        // Untracked = sum of (gt_frame_count - gt_tracked_count) for each GT
        let mut untracked = 0usize;
        for (&gt_id, &total_frames) in gt_frame_counts {
            let tracked = gt_tracked_counts.get(&gt_id).copied().unwrap_or(0);
            untracked += total_frames.saturating_sub(tracked);
        }

        result.metrics.untracked = untracked;
        result.metrics.total_pairs += untracked;
        result
    }

    /// Get all associations (for visualization)
    pub fn associations(&self) -> &[FrameAssociation] {
        &self.associations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_tracking() {
        let mut eval = Evaluator::new();

        // Tracker 1 always matches GT 1
        for frame in 0..10 {
            eval.add_association(FrameAssociation {
                frame_idx: frame,
                tracker_id: 1,
                gt_track_id: Some(1),
            });
        }

        let result = eval.compute_association_score();
        let m = &result.metrics;
        assert_eq!(m.matched, 10);
        assert_eq!(m.false_match, 0);
        assert_eq!(m.tracked_no_object, 0);
        assert_eq!(m.fragmentation, 0.0); // No ID switches
        assert_eq!(m.confusion, 0.0); // No bad switches
        assert!((m.coverage() - 1.0).abs() < 1e-6); // 10 / (10 + 0 + 0) = 1.0
        // Check assignment
        assert_eq!(result.assignment.get(&1), Some(&1));
    }

    #[test]
    fn test_id_switch() {
        let mut eval = Evaluator::new();

        // Tracker 1 matches GT 1 for 7 frames, then GT 2 for 3 frames
        // Assignment: tracker 1 -> GT 1 (by frequency)
        for frame in 0..7 {
            eval.add_association(FrameAssociation {
                frame_idx: frame,
                tracker_id: 1,
                gt_track_id: Some(1),
            });
        }
        for frame in 7..10 {
            eval.add_association(FrameAssociation {
                frame_idx: frame,
                tracker_id: 1,
                gt_track_id: Some(2),
            });
        }

        let result = eval.compute_association_score();
        let m = &result.metrics;
        // Tracker 1 assigned to GT 1 (7 > 3)
        // TP = 7 (frames 0-6 where tracker 1 on GT 1)
        // FP = 3 (frames 7-9 where tracker 1 on GT 2 but assigned to GT 1)
        assert_eq!(m.matched, 7);
        assert_eq!(m.false_match, 3);
        assert_eq!(result.assignment.get(&1), Some(&1));
    }

    #[test]
    fn test_tracker_on_non_gt() {
        let mut eval = Evaluator::new();

        // Tracker 1 on GT 1, tracker 2 on non-GT detection
        eval.add_association(FrameAssociation {
            frame_idx: 0,
            tracker_id: 1,
            gt_track_id: Some(1),
        });
        eval.add_association(FrameAssociation {
            frame_idx: 0,
            tracker_id: 2,
            gt_track_id: None, // no GT
        });

        let result = eval.compute_association_score();
        let m = &result.metrics;
        assert_eq!(m.matched, 1);
        assert_eq!(m.false_match, 0);
        assert_eq!(m.tracked_no_object, 1); // tracker 2 not assigned to any GT
    }

    #[test]
    fn test_fragmentation_with_switches() {
        let mut eval = Evaluator::new();

        // GT 1 tracked by tracker 1 for frames 0-4, then tracker 2 for frames 5-9
        // Tracker 1 assigned to GT 1 (5 frames), tracker 2 assigned to GT 1 (5 frames)
        // Both trackers are assigned to GT 1, so switching between them = 0 bad switches (confusion)
        // But fragmentation counts ALL switches, so fragmentation = 1 switch
        for frame in 0..5 {
            eval.add_association(FrameAssociation {
                frame_idx: frame,
                tracker_id: 1,
                gt_track_id: Some(1),
            });
        }
        for frame in 5..10 {
            eval.add_association(FrameAssociation {
                frame_idx: frame,
                tracker_id: 2,
                gt_track_id: Some(1),
            });
        }

        let result = eval.compute_association_score();
        let m = &result.metrics;
        // Both tracker 1 and tracker 2 are assigned to GT 1
        // Fragmentation = RMS of [1 switch] = 1.0
        // Confusion = 0 (switch is not to a bad pred_id)
        assert_eq!(m.fragmentation, 1.0);
        assert_eq!(m.confusion, 0.0);
    }

    #[test]
    fn test_fragmentation_bad_switches() {
        let mut eval = Evaluator::new();

        // GT 1 tracked by tracker 1 for frames 0-4, then tracker 2 for frames 5-9
        // GT 2 tracked by tracker 2 for frames 0-4 (so tracker 2 is assigned to GT 2)
        // When GT 1 switches to tracker 2 (assigned to GT 2), that's a bad switch
        for frame in 0..5 {
            eval.add_association(FrameAssociation { frame_idx: frame, tracker_id: 1, gt_track_id: Some(1) });
            eval.add_association(FrameAssociation { frame_idx: frame, tracker_id: 2, gt_track_id: Some(2) });
        }
        for frame in 5..10 {
            eval.add_association(FrameAssociation { frame_idx: frame, tracker_id: 2, gt_track_id: Some(1) });
        }

        let result = eval.compute_association_score();
        let m = &result.metrics;
        // Tracker 1 -> GT 1, Tracker 2 -> GT 2 (by frequency: tracker 2 has 5 GT2, 5 GT1, tie broken by... first seen?)
        // Actually tracker 2 sees GT2 5 times and GT1 5 times - it's a tie
        // Let's check assignment - with HashMap iteration order it might pick either
        // GT 1: switch from tracker 1 (assigned GT1) to tracker 2 (assigned GT2) = 1 bad switch
        // GT 2: no switches
        // Fragmentation = sqrt(mean of [1, 0]) = sqrt(0.5) ≈ 0.707
        // But if tracker 2 gets assigned to GT 1 instead, then 0 bad switches for GT 1
        // This test is tricky due to tie-breaking, let's make it clearer
    }

    #[test]
    fn test_fragmentation_clear_bad_switch() {
        let mut eval = Evaluator::new();

        // GT 1: tracker 1 (frames 0-6), tracker 2 (frames 7-9)
        // GT 2: tracker 2 (frames 0-6)
        // Tracker 1 -> GT 1 (7 frames), Tracker 2 -> GT 2 (7 frames > 3 frames on GT 1)
        // Switch from tracker 1 to tracker 2 on GT 1 is a bad switch
        for frame in 0..7 {
            eval.add_association(FrameAssociation { frame_idx: frame, tracker_id: 1, gt_track_id: Some(1) });
            eval.add_association(FrameAssociation { frame_idx: frame, tracker_id: 2, gt_track_id: Some(2) });
        }
        for frame in 7..10 {
            eval.add_association(FrameAssociation { frame_idx: frame, tracker_id: 2, gt_track_id: Some(1) });
        }

        let result = eval.compute_association_score();
        let m = &result.metrics;
        // Tracker 1 assigned to GT 1, Tracker 2 assigned to GT 2
        // GT 1: 1 switch (tracker 1 -> tracker 2), and it's a bad switch (tracker 2 is assigned to GT 2)
        // GT 2: 0 switches
        // Fragmentation = sqrt((1^2 + 0^2) / 2) = sqrt(0.5) ≈ 0.707
        // Confusion = sqrt((1^2 + 0^2) / 2) = sqrt(0.5) ≈ 0.707 (same because all switches are bad)
        assert!((m.fragmentation - 0.5_f32.sqrt()).abs() < 1e-5);
        assert!((m.confusion - 0.5_f32.sqrt()).abs() < 1e-5);
    }
}
