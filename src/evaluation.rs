//! Multi-Object Tracking (MOT) evaluation metrics.
//!
//! Implements MOTA (Multiple Object Tracking Accuracy) and HOTA (Higher Order Tracking Accuracy).

use std::collections::{HashMap, HashSet};

/// A ground truth detection for a single frame
#[derive(Debug, Clone)]
pub struct GtDetection {
    pub frame_idx: usize,
    pub track_id: u64,
    /// Bounding box in pixel coordinates: (x, y, w, h)
    pub bbox: (f32, f32, f32, f32),
}

/// A predicted track output for a single frame
#[derive(Debug, Clone)]
pub struct PredDetection {
    pub frame_idx: usize,
    pub track_id: u64,
    /// Bounding box in pixel coordinates: (x, y, w, h)
    pub bbox: (f32, f32, f32, f32),
}

/// Per-frame evaluation result for visualization
#[derive(Debug, Clone, Default)]
pub struct FrameEvalResult {
    /// True positives: (pred_track_id, gt_track_id, IoU)
    pub true_positives: Vec<(u64, u64, f32)>,
    /// False positives: pred_track_ids with no GT match
    pub false_positives: Vec<u64>,
    /// False negatives: gt_track_ids with no prediction match
    pub false_negatives: Vec<u64>,
    /// ID switches: (pred_track_id, old_gt_id, new_gt_id)
    pub id_switches: Vec<(u64, u64, u64)>,
}

/// Accumulated evaluation metrics
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub total_gt: usize,
    pub total_pred: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub id_switches: usize,
    /// DetA at various IoU thresholds
    pub det_a_by_threshold: Vec<(f32, f32)>,
    /// AssA at various IoU thresholds
    pub ass_a_by_threshold: Vec<(f32, f32)>,
}

impl EvalMetrics {
    /// MOTA = 1 - (FN + FP + IDSW) / GT
    pub fn mota(&self) -> f32 {
        if self.total_gt == 0 {
            return 0.0;
        }
        let errors = self.false_negatives + self.false_positives + self.id_switches;
        1.0 - (errors as f32 / self.total_gt as f32)
    }

    /// HOTA = sqrt(DetA * AssA), averaged over thresholds
    pub fn hota(&self) -> f32 {
        if self.det_a_by_threshold.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.det_a_by_threshold.iter()
            .zip(self.ass_a_by_threshold.iter())
            .map(|((_, det_a), (_, ass_a))| (det_a * ass_a).sqrt())
            .sum();
        sum / self.det_a_by_threshold.len() as f32
    }

    /// Detection accuracy (averaged over thresholds)
    pub fn det_a(&self) -> f32 {
        if self.det_a_by_threshold.is_empty() {
            return 0.0;
        }
        self.det_a_by_threshold.iter().map(|(_, v)| v).sum::<f32>()
            / self.det_a_by_threshold.len() as f32
    }

    /// Association accuracy (averaged over thresholds)
    pub fn ass_a(&self) -> f32 {
        if self.ass_a_by_threshold.is_empty() {
            return 0.0;
        }
        self.ass_a_by_threshold.iter().map(|(_, v)| v).sum::<f32>()
            / self.ass_a_by_threshold.len() as f32
    }

    /// IDF1 = 2*TP / (2*TP + FP + FN)
    pub fn idf1(&self) -> f32 {
        let denom = 2 * self.true_positives + self.false_positives + self.false_negatives;
        if denom == 0 {
            return 0.0;
        }
        (2 * self.true_positives) as f32 / denom as f32
    }

    /// Precision = TP / (TP + FP)
    pub fn precision(&self) -> f32 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f32 / denom as f32
    }

    /// Recall = TP / (TP + FN)
    pub fn recall(&self) -> f32 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f32 / denom as f32
    }
}

/// Evaluator for tracking performance
pub struct Evaluator {
    /// IoU threshold for matching (for MOTA)
    iou_threshold: f32,
    /// IoU thresholds for HOTA computation
    hota_thresholds: Vec<f32>,
    /// Track ID -> last matched GT ID (for detecting ID switches)
    pred_to_gt_mapping: HashMap<u64, u64>,
    /// Accumulated metrics
    metrics: EvalMetrics,
    /// Per-frame results for visualization
    frame_results: HashMap<usize, FrameEvalResult>,
    /// All GT detections grouped by frame
    gt_by_frame: HashMap<usize, Vec<GtDetection>>,
    /// All predictions grouped by frame
    pred_by_frame: HashMap<usize, Vec<PredDetection>>,
}

impl Evaluator {
    pub fn new(iou_threshold: f32) -> Self {
        // Standard HOTA thresholds: 0.05 to 0.95 in steps of 0.05
        let hota_thresholds: Vec<f32> = (1..=19).map(|i| i as f32 * 0.05).collect();

        Self {
            iou_threshold,
            hota_thresholds,
            pred_to_gt_mapping: HashMap::new(),
            metrics: EvalMetrics::default(),
            frame_results: HashMap::new(),
            gt_by_frame: HashMap::new(),
            pred_by_frame: HashMap::new(),
        }
    }

    /// Reset evaluator state
    pub fn reset(&mut self) {
        self.pred_to_gt_mapping.clear();
        self.metrics = EvalMetrics::default();
        self.frame_results.clear();
        self.gt_by_frame.clear();
        self.pred_by_frame.clear();
    }

    /// Add ground truth detections
    pub fn add_ground_truth(&mut self, detections: Vec<GtDetection>) {
        for det in detections {
            self.gt_by_frame.entry(det.frame_idx).or_default().push(det);
        }
    }

    /// Add predicted detections
    pub fn add_predictions(&mut self, detections: Vec<PredDetection>) {
        for det in detections {
            self.pred_by_frame.entry(det.frame_idx).or_default().push(det);
        }
    }

    /// Evaluate a single frame and return per-frame results
    pub fn evaluate_frame(&mut self, frame_idx: usize) -> FrameEvalResult {
        let gt_dets = self.gt_by_frame.get(&frame_idx).cloned().unwrap_or_default();
        let pred_dets = self.pred_by_frame.get(&frame_idx).cloned().unwrap_or_default();

        let mut result = FrameEvalResult::default();
        let mut matched_gt: HashSet<u64> = HashSet::new();
        let mut matched_pred: HashSet<u64> = HashSet::new();

        // Compute IoU matrix and find matches (greedy matching by highest IoU)
        let mut matches: Vec<(usize, usize, f32)> = Vec::new();
        for (pi, pred) in pred_dets.iter().enumerate() {
            for (gi, gt) in gt_dets.iter().enumerate() {
                let iou = compute_iou(&pred.bbox, &gt.bbox);
                if iou >= self.iou_threshold {
                    matches.push((pi, gi, iou));
                }
            }
        }

        // Sort by IoU descending and greedily assign
        matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        for (pi, gi, iou) in matches {
            let pred = &pred_dets[pi];
            let gt = &gt_dets[gi];

            if matched_pred.contains(&pred.track_id) || matched_gt.contains(&gt.track_id) {
                continue;
            }

            matched_pred.insert(pred.track_id);
            matched_gt.insert(gt.track_id);

            // Check for ID switch
            if let Some(&prev_gt_id) = self.pred_to_gt_mapping.get(&pred.track_id) {
                if prev_gt_id != gt.track_id {
                    result.id_switches.push((pred.track_id, prev_gt_id, gt.track_id));
                    self.metrics.id_switches += 1;
                }
            }
            self.pred_to_gt_mapping.insert(pred.track_id, gt.track_id);

            result.true_positives.push((pred.track_id, gt.track_id, iou));
            self.metrics.true_positives += 1;
        }

        // Unmatched predictions -> false positives
        for pred in &pred_dets {
            if !matched_pred.contains(&pred.track_id) {
                result.false_positives.push(pred.track_id);
                self.metrics.false_positives += 1;
            }
        }

        // Unmatched GT -> false negatives
        for gt in &gt_dets {
            if !matched_gt.contains(&gt.track_id) {
                result.false_negatives.push(gt.track_id);
                self.metrics.false_negatives += 1;
            }
        }

        self.metrics.total_gt += gt_dets.len();
        self.metrics.total_pred += pred_dets.len();

        self.frame_results.insert(frame_idx, result.clone());
        result
    }

    /// Compute HOTA metrics after all frames have been processed
    pub fn compute_hota(&mut self) {
        self.metrics.det_a_by_threshold.clear();
        self.metrics.ass_a_by_threshold.clear();

        for &threshold in &self.hota_thresholds.clone() {
            let (det_a, ass_a) = self.compute_hota_at_threshold(threshold);
            self.metrics.det_a_by_threshold.push((threshold, det_a));
            self.metrics.ass_a_by_threshold.push((threshold, ass_a));
        }
    }

    fn compute_hota_at_threshold(&self, threshold: f32) -> (f32, f32) {
        // Collect all matches at this threshold
        let mut tp_count = 0;
        let mut fp_count = 0;
        let mut fn_count = 0;

        // For AssA: track (pred_id, gt_id) pairs across frames
        let mut pred_gt_pairs: HashMap<(u64, u64), usize> = HashMap::new();
        let mut pred_totals: HashMap<u64, usize> = HashMap::new();
        let mut gt_totals: HashMap<u64, usize> = HashMap::new();

        // Re-evaluate at this threshold
        for frame_idx in self.gt_by_frame.keys().chain(self.pred_by_frame.keys()) {
            let gt_dets = self.gt_by_frame.get(frame_idx).cloned().unwrap_or_default();
            let pred_dets = self.pred_by_frame.get(frame_idx).cloned().unwrap_or_default();

            let mut matches: Vec<(usize, usize, f32)> = Vec::new();
            for (pi, pred) in pred_dets.iter().enumerate() {
                for (gi, gt) in gt_dets.iter().enumerate() {
                    let iou = compute_iou(&pred.bbox, &gt.bbox);
                    if iou >= threshold {
                        matches.push((pi, gi, iou));
                    }
                }
            }

            matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            let mut matched_pred: HashSet<usize> = HashSet::new();
            let mut matched_gt: HashSet<usize> = HashSet::new();

            for (pi, gi, _) in matches {
                if matched_pred.contains(&pi) || matched_gt.contains(&gi) {
                    continue;
                }
                matched_pred.insert(pi);
                matched_gt.insert(gi);

                let pred_id = pred_dets[pi].track_id;
                let gt_id = gt_dets[gi].track_id;

                tp_count += 1;
                *pred_gt_pairs.entry((pred_id, gt_id)).or_default() += 1;
            }

            fp_count += pred_dets.len() - matched_pred.len();
            fn_count += gt_dets.len() - matched_gt.len();

            for pred in &pred_dets {
                *pred_totals.entry(pred.track_id).or_default() += 1;
            }
            for gt in &gt_dets {
                *gt_totals.entry(gt.track_id).or_default() += 1;
            }
        }

        // DetA = TP / (TP + FP + FN)
        let det_denom = tp_count + fp_count + fn_count;
        let det_a = if det_denom > 0 {
            tp_count as f32 / det_denom as f32
        } else {
            0.0
        };

        // AssA: for each TP, compute |A(c)| / (|TPA(c)| + |FPA(c)| + |FNA(c)|)
        // where A(c) is the set of TPs with same (pred_id, gt_id) as this TP
        let mut ass_sum = 0.0f32;
        for ((pred_id, gt_id), count) in &pred_gt_pairs {
            let tpa = *count as f32;
            // FPA: other GTs matched to this pred
            let total_pred_matches: usize = pred_gt_pairs
                .iter()
                .filter(|((p, _), _)| p == pred_id)
                .map(|(_, c)| c)
                .sum();
            let fpa = total_pred_matches as f32 - tpa;

            // FNA: other preds matched to this GT
            let total_gt_matches: usize = pred_gt_pairs
                .iter()
                .filter(|((_, g), _)| g == gt_id)
                .map(|(_, c)| c)
                .sum();
            let fna = total_gt_matches as f32 - tpa;

            let ass_denom = tpa + fpa + fna;
            if ass_denom > 0.0 {
                ass_sum += (*count as f32) * (tpa / ass_denom);
            }
        }

        let ass_a = if tp_count > 0 {
            ass_sum / tp_count as f32
        } else {
            0.0
        };

        (det_a, ass_a)
    }

    /// Get accumulated metrics
    pub fn metrics(&self) -> &EvalMetrics {
        &self.metrics
    }

    /// Get per-frame results for visualization
    pub fn frame_result(&self, frame_idx: usize) -> Option<&FrameEvalResult> {
        self.frame_results.get(&frame_idx)
    }

    /// Get all frame indices that have been evaluated
    pub fn evaluated_frames(&self) -> Vec<usize> {
        let mut frames: Vec<usize> = self.frame_results.keys().copied().collect();
        frames.sort();
        frames
    }
}

/// Compute IoU between two bounding boxes (x, y, w, h)
fn compute_iou(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> f32 {
    let (ax, ay, aw, ah) = *a;
    let (bx, by, bw, bh) = *b;

    let ax2 = ax + aw;
    let ay2 = ay + ah;
    let bx2 = bx + bw;
    let by2 = by + bh;

    let inter_x1 = ax.max(bx);
    let inter_y1 = ay.max(by);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = aw * ah;
    let area_b = bw * bh;
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou() {
        // Same box
        let a = (0.0, 0.0, 10.0, 10.0);
        assert!((compute_iou(&a, &a) - 1.0).abs() < 1e-6);

        // No overlap
        let b = (20.0, 20.0, 10.0, 10.0);
        assert!(compute_iou(&a, &b) < 1e-6);

        // 50% overlap
        let c = (5.0, 0.0, 10.0, 10.0);
        // Intersection: 5x10 = 50, Union: 100 + 100 - 50 = 150
        let iou = compute_iou(&a, &c);
        assert!((iou - 50.0 / 150.0).abs() < 1e-5);
    }

    #[test]
    fn test_perfect_tracking() {
        let mut evaluator = Evaluator::new(0.5);

        evaluator.add_ground_truth(vec![
            GtDetection { frame_idx: 0, track_id: 1, bbox: (0.0, 0.0, 10.0, 10.0) },
            GtDetection { frame_idx: 1, track_id: 1, bbox: (1.0, 0.0, 10.0, 10.0) },
        ]);

        evaluator.add_predictions(vec![
            PredDetection { frame_idx: 0, track_id: 1, bbox: (0.0, 0.0, 10.0, 10.0) },
            PredDetection { frame_idx: 1, track_id: 1, bbox: (1.0, 0.0, 10.0, 10.0) },
        ]);

        evaluator.evaluate_frame(0);
        evaluator.evaluate_frame(1);
        evaluator.compute_hota();

        let metrics = evaluator.metrics();
        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.id_switches, 0);
        assert!((metrics.mota() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_id_switch() {
        let mut evaluator = Evaluator::new(0.5);

        evaluator.add_ground_truth(vec![
            GtDetection { frame_idx: 0, track_id: 1, bbox: (0.0, 0.0, 10.0, 10.0) },
            GtDetection { frame_idx: 0, track_id: 2, bbox: (20.0, 0.0, 10.0, 10.0) },
            GtDetection { frame_idx: 1, track_id: 1, bbox: (20.0, 0.0, 10.0, 10.0) }, // GT 1 moved to where 2 was
            GtDetection { frame_idx: 1, track_id: 2, bbox: (0.0, 0.0, 10.0, 10.0) },  // GT 2 moved to where 1 was
        ]);

        // Predictions stay in place (causing ID switch)
        evaluator.add_predictions(vec![
            PredDetection { frame_idx: 0, track_id: 1, bbox: (0.0, 0.0, 10.0, 10.0) },
            PredDetection { frame_idx: 0, track_id: 2, bbox: (20.0, 0.0, 10.0, 10.0) },
            PredDetection { frame_idx: 1, track_id: 1, bbox: (0.0, 0.0, 10.0, 10.0) },  // Pred 1 matches GT 2 now
            PredDetection { frame_idx: 1, track_id: 2, bbox: (20.0, 0.0, 10.0, 10.0) }, // Pred 2 matches GT 1 now
        ]);

        evaluator.evaluate_frame(0);
        evaluator.evaluate_frame(1);

        let metrics = evaluator.metrics();
        assert_eq!(metrics.true_positives, 4);
        assert_eq!(metrics.id_switches, 2);
    }
}
