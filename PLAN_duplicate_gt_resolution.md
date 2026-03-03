# Duplicate GT ID Resolution - Work in Progress

## Current State

### What's Done
1. **Timeline UI improvements** (committed ready):
   - Moved time series plot from video overlay to timeline panel
   - Timeline panel is now 200px when eval overlays active
   - Click-to-seek on the time series plot
   - Removed old `draw_gt_timeseries` from drawing.rs

2. **IoU-based duplicate resolution** (in dataset.rs - TO BE REPLACED):
   - `resolve_duplicate_gt_ids()` currently uses IoU to track duplicates
   - Keeps longest pseudo-track, nullifies others
   - Called in both evaluate.rs and visualizer/main.rs after loading

### Files Modified (uncommitted)
- `src/dataset.rs` - Has IoU-based `resolve_duplicate_gt_ids()` (will be replaced)
- `src/bin/evaluate.rs` - Calls `clip.resolve_duplicate_gt_ids()` after loading
- `src/bin/visualizer/main.rs` - Calls `clip.resolve_duplicate_gt_ids()` after loading
- `src/bin/visualizer/ui.rs` - Timeline improvements, time series in panel
- `src/bin/visualizer/drawing.rs` - Removed `draw_gt_timeseries`

## Next Steps: Pred-ID Based Resolution

### Problem
Current IoU-based approach guesses which duplicate detection is "correct" at load time.
Better approach: Let ByteTrack decide, then retroactively keep the GT labels that align with tracker decisions.

### Planned Algorithm
1. **Run ByteTrack first** (as usual)
2. **For each GT ID with duplicates:**
   - Collect all detections with that GT ID across all frames
   - Look at which `pred_id` the tracker assigned to each
   - Group detections by their assigned `pred_id`
   - Find the pred_id group that:
     - Contains >50% of total detections with this GT label
     - Has minimal track switches (ideally single contiguous pred_id)
   - Keep only those detections with their `gt_track_id`
   - Set `gt_track_id = None` for all others
3. **Re-compute evaluation metrics** on cleaned data

### Implementation Plan

**New shared function in `evaluation.rs`:**
```rust
/// Given associations from tracker, resolve duplicate GT IDs by keeping
/// only the pred_id sequence that covers >50% and minimizes switches.
///
/// Input: Vec<(frame_idx, det_idx, pred_id, gt_track_id)>
/// Output: Set of (frame_idx, det_idx) that should have gt_track_id nullified
pub fn resolve_duplicate_gt_ids_by_pred(
    associations: &[(usize, usize, u64, Option<u64>)],
) -> (HashSet<(usize, usize)>, usize, usize)  // (to_nullify, num_dups, num_nullified)
```

**Changes needed:**
1. Add `resolve_duplicate_gt_ids_by_pred()` to `src/evaluation.rs`
2. Remove `resolve_duplicate_gt_ids()` from `src/dataset.rs`
3. Update `evaluate.rs`:
   - Remove call to `clip.resolve_duplicate_gt_ids()`
   - After tracking loop, build associations list
   - Call `resolve_duplicate_gt_ids_by_pred()`
   - Apply nullifications before computing metrics
4. Update `visualizer/main.rs`:
   - Remove call to `clip.resolve_duplicate_gt_ids()`
   - In `run_tracker()`, after tracking, build associations
   - Call shared function, apply nullifications to `clip.detections_by_frame`
   - Continue with evaluation as before

### Key Insight
The tracker has already done the hard work of associating detections across frames.
If GT ID 5 appears twice in frame N but tracker assigned different pred_ids,
we use that to decide which detection is the "real" GT 5 vs a duplicate/occlusion.
