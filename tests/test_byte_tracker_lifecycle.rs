use jamtrack_rs::byte_tracker::{ByteTracker, TrackBufferSizes};
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

/* ----------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------------- */

/// Creates a ByteTracker with default parameters.
/// max_time_lost = track_buffer * frame_rate / 30 = 30
fn default_byte_tracker() -> ByteTracker {
    ByteTracker::new(
        30,        // frame_rate
        30,        // track_buffer
        0.5,       // track_thresh
        0.7,       // high_thresh
        false,     // use_ciou
        0.3,       // high_conf_match_iou_weight
        1.0,       // high_conf_match_min_iou
        0.5,       // low_conf_match_iou_weight
        1.0,       // low_conf_match_min_iou
        0.3,       // track_activation_iou_weight
        1.0,       // track_activation_min_iou
        1. / 20.,  // kalman_std_weight_pos
        1. / 160., // kalman_std_weight_vel
        1. / 20.,  // kalman_std_weight_position_meas
        1. / 20.,  // kalman_std_weight_position_mot
        1. / 160., // kalman_std_weight_velocity_mot
        1e-2,      // kalman_std_aspect_ratio_init
        1e-5,      // kalman_std_d_aspect_ratio_init
        1e-2,      // kalman_std_aspect_ratio_mot
        1e-5,      // kalman_std_d_aspect_ratio_mot
        1e-1,      // kalman_std_aspect_ratio_meas
    )
}

/// Creates a ByteTracker with a short TTL for testing lost/removed transitions.
/// max_time_lost = track_buffer * frame_rate / 30 = 5 * 30 / 30 = 5
fn short_ttl_byte_tracker() -> ByteTracker {
    ByteTracker::new(
        30,        // frame_rate
        5,         // track_buffer (short TTL → max_time_lost = 5)
        0.5,       // track_thresh
        0.7,       // high_thresh
        false,     // use_ciou
        0.3,       // high_conf_match_iou_weight
        1.0,       // high_conf_match_min_iou
        0.5,       // low_conf_match_iou_weight
        1.0,       // low_conf_match_min_iou
        0.3,       // track_activation_iou_weight
        1.0,       // track_activation_min_iou
        1. / 20.,  // kalman_std_weight_pos
        1. / 160., // kalman_std_weight_vel
        1. / 20.,  // kalman_std_weight_position_meas
        1. / 20.,  // kalman_std_weight_position_mot
        1. / 160., // kalman_std_weight_velocity_mot
        1e-2,      // kalman_std_aspect_ratio_init
        1e-5,      // kalman_std_d_aspect_ratio_init
        1e-2,      // kalman_std_aspect_ratio_mot
        1e-5,      // kalman_std_d_aspect_ratio_mot
        1e-1,      // kalman_std_aspect_ratio_meas
    )
}

fn make_det(id: i64, x: f32, y: f32, w: f32, h: f32, prob: f32) -> Object {
    Object::new(id, Rect::new(x, y, w, h), prob, None, None)
}

fn empty_frame() -> std::iter::Empty<Object> {
    std::iter::empty()
}

/* ----------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------------- */

/// A single high-confidence detection on the first frame should immediately
/// spawn and activate a track.
#[test]
fn test_single_high_conf_detection_spawns_track() {
    let mut tracker = default_byte_tracker();

    let det = make_det(1, 100.0, 100.0, 50.0, 50.0, 0.9);
    let output = tracker.update(vec![det].into_iter()).unwrap();

    assert_eq!(output.len(), 1, "expected exactly one output track");
    assert_eq!(output[0].get_track_id(), Some(1));
    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 1,
            lost: 0,
            removed: 0,
        }
    );
}

/// Feeding the same high-confidence detection at the same location across
/// multiple frames should produce exactly one track (the detections associate
/// to it rather than spawning new tracks).
#[test]
fn test_static_high_conf_detections_spawn_single_track() {
    let mut tracker = default_byte_tracker();

    for frame in 0..10 {
        let det = make_det(frame as i64, 100.0, 100.0, 50.0, 50.0, 0.9);
        let output = tracker.update(vec![det].into_iter()).unwrap();

        assert_eq!(output.len(), 1, "frame {}: expected 1 output track", frame + 1);
        assert_eq!(
            output[0].get_track_id(),
            Some(1),
            "frame {}: expected track_id 1",
            frame + 1
        );
    }

    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 1,
            lost: 0,
            removed: 0,
        }
    );
}

/// A high-confidence detection on frame 1 spawns a track. Subsequent frames
/// provide only low-confidence detections (below track_thresh but above 0) at
/// the same location. The low-conf dets should associate to the existing track
/// via the second-stage matching, keeping exactly one track alive.
#[test]
fn test_high_conf_then_low_conf_single_track() {
    let mut tracker = default_byte_tracker();

    // Frame 1: high-confidence detection spawns and activates the track.
    let det = make_det(0, 100.0, 100.0, 50.0, 50.0, 0.9);
    let output = tracker.update(vec![det].into_iter()).unwrap();
    assert_eq!(output.len(), 1);
    let track_id = output[0].get_track_id().unwrap();

    // Frames 2-10: low-confidence detections at the same location.
    // prob = 0.4 is below track_thresh (0.5) so these go to the low-conf pool.
    for frame in 1..10 {
        let det = make_det(frame as i64, 100.0, 100.0, 50.0, 50.0, 0.4);
        let output = tracker.update(vec![det].into_iter()).unwrap();

        assert_eq!(
            output.len(),
            1,
            "frame {}: expected 1 output track",
            frame + 1
        );
        assert_eq!(
            output[0].get_track_id(),
            Some(track_id),
            "frame {}: should still be the same track",
            frame + 1
        );
    }

    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 1,
            lost: 0,
            removed: 0,
        }
    );
}

/// A high-confidence detection spawns a track. When no further detections are
/// provided the track should transition: Tracked → Lost → Removed.
///
/// Uses short_ttl_byte_tracker (max_time_lost = 5).
///   - Frame 1: track spawned and activated
///   - Frame 2: track becomes lost (unmatched in both association stages)
///   - Frames 3-6: track stays lost (TTL not yet expired)
///   - Frame 7: track removed (frame_id(7) - track.frame_id(1) = 6 > 5)
#[test]
fn test_track_becomes_lost_then_removed() {
    let mut tracker = short_ttl_byte_tracker();

    // Frame 1: spawn track
    let det = make_det(0, 100.0, 100.0, 50.0, 50.0, 0.9);
    let output = tracker.update(vec![det].into_iter()).unwrap();
    assert_eq!(output.len(), 1, "track should be active on frame 1");
    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 1,
            lost: 0,
            removed: 0,
        }
    );

    // Frame 2: no detections → track becomes lost
    let output = tracker.update(empty_frame()).unwrap();
    assert_eq!(output.len(), 0, "no active tracks on frame 2");
    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 0,
            lost: 1,
            removed: 0,
        }
    );

    // Update 13 more frames; at some point the lost track's TTL expires
    // and it transitions from lost to removed.
    let mut removal_frame = None;
    for frame in 3..=15 {
        let output = tracker.update(empty_frame()).unwrap();
        assert_eq!(output.len(), 0, "no active tracks on frame {}", frame);

        let sizes = tracker.track_buffer_sizes();
        assert_eq!(sizes.tracked, 0, "frame {}: no tracked", frame);

        if sizes.removed == 1 {
            removal_frame = Some(frame);
        }
    }
    let removal_frame = removal_frame.expect("track should have been removed within 15 frames");
    assert_eq!(tracker.track_buffer_sizes().lost, 0, "there should be no more lost tracks");

    // Removed tracks are only reported on the frame they are removed, not accumulated.
    // Run one more frame past the removal and verify the removed buffer is empty.
    let output = tracker.update(empty_frame()).unwrap();
    assert_eq!(output.len(), 0);
    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 0,
            lost: 0,
            removed: 0,
        },
        "removed tracks should not persist beyond the frame they were removed (removal was frame {})",
        removal_frame
    );
}

/// A high-confidence detection spawns a track. The track becomes lost for
/// several frames but is revived by a new detection before the TTL expires.
/// The revived track should keep its original track_id.
///
/// Uses short_ttl_byte_tracker (max_time_lost = 5).
///   - Frame 1: track spawned and activated
///   - Frames 2-4: no detections → track is lost
///   - Frame 5: high-conf detection at same location → track revived
#[test]
fn test_track_lost_then_revived_before_ttl() {
    let mut tracker = short_ttl_byte_tracker();

    // Frame 1: spawn track
    let det = make_det(0, 100.0, 100.0, 50.0, 50.0, 0.9);
    let output = tracker.update(vec![det].into_iter()).unwrap();
    assert_eq!(output.len(), 1);
    let original_track_id = output[0].get_track_id().unwrap();

    // Frame 2: no detection → lost
    let output = tracker.update(empty_frame()).unwrap();
    assert_eq!(output.len(), 0);
    assert_eq!(tracker.track_buffer_sizes().lost, 1);

    // Frames 3-4: still no detections → still lost
    for _ in 3..=4 {
        let output = tracker.update(empty_frame()).unwrap();
        assert_eq!(output.len(), 0);
    }
    let sizes = tracker.track_buffer_sizes();
    assert_eq!(sizes.lost, 1, "track should still be lost before revival");
    assert_eq!(sizes.removed, 0, "track should not be removed yet");

    // Frame 5: revive with a new detection at the same location
    let det = make_det(1, 100.0, 100.0, 50.0, 50.0, 0.9);
    let output = tracker.update(vec![det].into_iter()).unwrap();

    assert_eq!(output.len(), 1, "revived track should appear in output");
    assert_eq!(
        output[0].get_track_id(),
        Some(original_track_id),
        "revived track should keep original track_id"
    );
    assert_eq!(
        tracker.track_buffer_sizes(),
        TrackBufferSizes {
            tracked: 1,
            lost: 0,
            removed: 0,
        }
    );
}