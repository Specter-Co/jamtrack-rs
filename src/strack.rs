use crate::{
    kalman_filter::{KalmanFilter, StateCov, StateMean},
    rect::Rect,
};
use core::time::Duration;
use std::fmt::Debug;

/* ----------------------------------------------------------------------------
 * STrack State enums
 * ---------------------------------------------------------------------------- */
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum STrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

/* ----------------------------------------------------------------------------
 * STrack struct
 * ---------------------------------------------------------------------------- */

impl Debug for STrack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "STrack {{ track_id: {}, timestamp: {:?}, start_timestamp: {:?}, tracklet_len: {}, state: {:?}, is_activated: {}, score: {}, rect: {:?} }}",
            self.track_id, self.timestamp, self.start_timestamp, self.tracklet_len, self.state, self.is_activated, self.score, self.rect
        )
    }
}

#[derive(Clone)]
pub(crate) struct STrack {
    kalman_filter: KalmanFilter,
    mean: StateMean,
    covariance: StateCov,
    rect: Rect<f32>,
    state: STrackState,
    is_activated: bool,
    score: f32,
    track_id: usize,
    timestamp: Duration,
    detection_id_last: i64,
    start_timestamp: Duration,
    tracklet_len: usize,
}

impl STrack {
    pub(crate) fn new(detection_id: i64, rect: Rect<f32>, score: f32) -> Self {
        let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160.);
        let mean = StateMean::zeros();
        let covariance = StateCov::zeros();
        Self {
            kalman_filter,
            mean,
            covariance,
            rect,
            state: STrackState::New,
            is_activated: false,
            score,
            track_id: 0,
            timestamp: Default::default(),
            detection_id_last: detection_id,
            start_timestamp: Default::default(),
            tracklet_len: 0,
        }
    }

    // This function is used in the test_joint_strack function in src/test_byte_tracker.rs
    #[cfg(test)]
    pub(crate) fn dummy_strack(track_id: usize) -> Self {
        let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160.);
        let mean = StateMean::zeros();
        let covariance = StateCov::zeros();
        Self {
            kalman_filter,
            mean,
            covariance,
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            state: STrackState::New,
            is_activated: false,
            score: 0.0,
            track_id: track_id,
            timestamp: 0,
            start_timestamp: 0,
            tracklet_len: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn get_detection_id_last(&self) -> i64 {
        self.detection_id_last
    }

    #[inline(always)]
    pub(crate) fn get_rect(&self) -> Rect<f32> {
        return self.rect.clone();
    }

    #[inline(always)]
    pub(crate) fn get_strack_state(&self) -> STrackState {
        return self.state;
    }

    #[inline(always)]
    pub(crate) fn is_activated(&self) -> bool {
        return self.is_activated;
    }

    #[inline(always)]
    pub(crate) fn get_score(&self) -> f32 {
        return self.score;
    }

    #[inline(always)]
    pub(crate) fn get_track_id(&self) -> usize {
        return self.track_id;
    }

    #[inline(always)]
    pub(crate) fn get_timestamp(&self) -> Duration {
        return self.timestamp;
    }

    #[inline(always)]
    pub(crate) fn get_start_timestamp(&self) -> Duration {
        return self.start_timestamp;
    }

    #[inline(always)]
    pub(crate) fn get_vel_x(&self) -> f32 {
        return self.mean[4];
    }

    #[inline(always)]
    pub(crate) fn get_vel_y(&self) -> f32 {
        return self.mean[5];
    }

    pub(crate) fn activate(&mut self, timestamp: Duration, track_id: usize) {
        self.kalman_filter.initiate(
            &mut self.mean,
            &mut self.covariance,
            &self.rect.get_xyah(),
        );

        self.update_rect();

        self.state = STrackState::Tracked;
        if timestamp == 1 {
            self.is_activated = true;
        }
        self.track_id = track_id;
        self.timestamp = timestamp;
        self.start_timestamp = timestamp;
        self.tracklet_len = 0;
    }

    pub(crate) fn re_activate(
        &mut self,
        new_track: &STrack,
        timestamp: Duration,
        new_track_id: isize,
    ) {
        self.kalman_filter.update(
            &mut self.mean,
            &mut self.covariance,
            &new_track.get_rect().get_xyah(),
        );
        self.update_rect();

        self.state = STrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.get_score();

        if 0 <= new_track_id {
            self.track_id = new_track_id as usize;
        }

        self.detection_id_last = new_track.detection_id_last;
        self.timestamp = timestamp;
        self.tracklet_len = 0;
    }

    pub(crate) fn predict(&mut self) {
        if self.state != STrackState::Tracked {
            self.mean[(0, 7)] = 0.;
        }
        self.kalman_filter
            .predict(&mut self.mean, &mut self.covariance);
        self.update_rect();
    }

    pub(crate) fn update(&mut self, new_track: &STrack, timestamp: Duration) {
        self.kalman_filter.update(
            &mut self.mean,
            &mut self.covariance,
            &new_track.get_rect().get_xyah(),
        );

        self.update_rect();

        self.state = STrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.get_score();
        self.timestamp = timestamp;
        self.detection_id_last = new_track.get_detection_id_last();
        self.tracklet_len += 1;
    }

    pub(crate) fn mark_as_lost(&mut self) {
        self.state = STrackState::Lost;
    }

    pub(crate) fn mark_as_removed(&mut self) {
        self.state = STrackState::Removed;
    }

    pub(crate) fn update_rect(&mut self) {
        self.rect.set_width(self.mean[(0, 2)] * self.mean[(0, 3)]);
        self.rect.set_height(self.mean[(0, 3)]);
        self.rect.set_x(self.mean[(0, 0)] - self.rect.width() / 2.);
        self.rect.set_y(self.mean[(0, 1)] - self.rect.height() / 2.);
    }
}

impl PartialEq for STrack {
    fn eq(&self, other: &Self) -> bool {
        return self.track_id == other.track_id;
    }
}
