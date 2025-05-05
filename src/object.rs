use crate::rect::Rect;
use crate::strack::STrack;
use std::fmt::Debug;

/* ------------------------------------------------------------------------------
 * Object struct
 * ------------------------------------------------------------------------------ */

#[derive(Debug, Clone)]
pub struct Object {
    detection_id: i64,
    prob: f32,
    rect: Rect<f32>,
    track_id: Option<usize>,
    track_vel_xy: Option<(f32, f32)>,
}

impl Object {
    pub fn new(
        detection_id: i64,
        rect: Rect<f32>,
        prob: f32,
        track_id: Option<usize>,
        track_vel_xy: Option<(f32, f32)>,
    ) -> Self {
        Self {
            detection_id,
            prob,
            rect,
            track_id,
            track_vel_xy,
        }
    }

    #[inline(always)]
    pub fn get_detection_id(&self) -> i64 {
        self.detection_id
    }

    #[inline(always)]
    pub fn get_rect(&self) -> Rect<f32> {
        self.rect.clone()
    }

    #[inline(always)]
    pub fn get_x(&self) -> f32 {
        self.rect.x()
    }

    #[inline(always)]
    pub fn get_y(&self) -> f32 {
        self.rect.y()
    }

    #[inline(always)]
    pub fn get_width(&self) -> f32 {
        self.rect.width()
    }

    #[inline(always)]
    pub fn get_height(&self) -> f32 {
        self.rect.height()
    }

    #[inline(always)]
    pub fn get_prob(&self) -> f32 {
        self.prob
    }

    #[inline(always)]
    pub fn get_track_id(&self) -> Option<usize> {
        self.track_id
    }

    #[inline(always)]
    pub fn get_track_vel_xy(&self) -> Option<(f32, f32)> {
        self.track_vel_xy
    }
}

impl From<STrack> for Object {
    fn from(strack: STrack) -> Self {
        Object::new(
            strack.get_detection_id_last(),
            strack.get_rect(),
            strack.get_score(),
            Some(strack.get_track_id()),
            Some((strack.get_vel_x(), strack.get_vel_y())),
        )
    }
}

impl From<&STrack> for Object {
    fn from(strack: &STrack) -> Self {
        Object::new(
            strack.get_detection_id_last(),
            strack.get_rect(),
            strack.get_score(),
            Some(strack.get_track_id()),
            Some((strack.get_vel_x(), strack.get_vel_y())),
        )
    }
}
