use pyo3::prelude::*;
use crate::rect::PyRect;
use crate::strack::STrack;
use std::fmt::Debug;

/* ------------------------------------------------------------------------------
 * Object struct
 * ------------------------------------------------------------------------------ */
 
 #[pyclass]
#[derive(Debug, Clone)]
pub struct Object {
    #[pyo3(get)]
    detection_id: i64,
    #[pyo3(get)]
    prob: f32,
    #[pyo3(get)]
    rect: PyRect,
    #[pyo3(get)]
    track_id: Option<usize>,
    #[pyo3(get)]
    track_vel_xy: Option<(f32, f32)>,
}

#[pymethods]
impl Object {
    #[new]
    pub fn new(
        detection_id: i64,
        rect: PyRect,
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

    #[getter]
    #[inline(always)]
    pub fn get_detection_id(&self) -> i64 {
        self.detection_id
    }

    #[getter]
    #[inline(always)]
    pub fn get_rect(&self) -> PyRect {
        self.rect.clone()
    }

    #[getter]
    #[inline(always)]
    pub fn get_x(&self) -> f32 {
        self.rect.x()
    }

    #[getter]
    #[inline(always)]
    pub fn get_y(&self) -> f32 {
        self.rect.y()
    }

    #[getter]
    #[inline(always)]
    pub fn get_width(&self) -> f32 {
        self.rect.width()
    }

    #[getter]
    #[inline(always)]
    pub fn get_height(&self) -> f32 {
        self.rect.height()
    }

    #[getter]
    #[inline(always)]
    pub fn get_prob(&self) -> f32 {
        self.prob
    }

    #[getter]
    #[inline(always)]
    pub fn get_track_id(&self) -> Option<usize> {
        self.track_id
    }

    #[getter]
    #[inline(always)]
    pub fn get_track_vel_xy(&self) -> Option<(f32, f32)> {
        self.track_vel_xy
    }
}

impl From<STrack> for Object {
    fn from(strack: STrack) -> Self {
        Object::new(
            strack.get_detection_id_last(),
            strack.get_rect().into(),
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
            strack.get_rect().into(),
            strack.get_score(),
            Some(strack.get_track_id()),
            Some((strack.get_vel_x(), strack.get_vel_y())),
        )
    }
}
