use pyo3::prelude::*;
use nalgebra::Matrix1x4;
use num::Float;
use std::fmt::Debug;

/* ------------------------------------------------------------------------------
 * Type aliases
 * ------------------------------------------------------------------------------ */
// type Tlwh<T> = Matrix1x4<T>;

type Xyah<T> = Matrix1x4<T>;

/* ------------------------------------------------------------------------------
 * Rect struct
 * ------------------------------------------------------------------------------ */
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rect<T>
where
    T: Debug + Float,
{
    tlwh: Matrix1x4<T>,
}

// Wrapper struct for Python
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRect {
    inner: Rect<f32>,
}

#[pymethods]
impl PyRect {
    #[new]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        let inner = Rect::new(x, y, width, height);
        Self { inner }
    }

    #[getter]
    pub fn x(&self) -> f32 {
        self.inner.x()
    }

    #[setter]
    pub fn set_x(&mut self, x: f32) {
        self.inner.set_x(x);
    }

    #[getter]
    pub fn y(&self) -> f32 {
        self.inner.y()
    }

    #[setter]
    pub fn set_y(&mut self, y: f32) {
        self.inner.set_y(y);
    }

    #[getter]
    pub fn width(&self) -> f32 {
        self.inner.width()
    }

    #[setter]
    pub fn set_width(&mut self, width: f32) {
        self.inner.set_width(width);
    }

    #[getter]
    pub fn height(&self) -> f32 {
        self.inner.height()
    }

    #[setter]
    pub fn set_height(&mut self, height: f32) {
        self.inner.set_height(height);
    }

    pub fn area(&self) -> f32 {
        self.inner.area()
    }
}

impl<T> Rect<T>
where
    T: Clone + Debug + Float,
{
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        let tlwh =
            Matrix1x4::new(x.clone(), y.clone(), width.clone(), height.clone());
        Self { tlwh }
    }

    #[inline(always)]
    pub fn x(&self) -> T {
        self.tlwh[(0, 0)]
    }

    #[inline(always)]
    pub(crate) fn set_x(&mut self, x: T) {
        self.tlwh[(0, 0)] = x;
    }

    #[inline(always)]
    pub fn y(&self) -> T {
        self.tlwh[(0, 1)]
    }

    #[inline(always)]
    pub(crate) fn set_y(&mut self, y: T) {
        self.tlwh[(0, 1)] = y;
    }

    #[inline(always)]
    pub fn width(&self) -> T {
        self.tlwh[(0, 2)]
    }

    #[inline(always)]
    pub(crate) fn set_width(&mut self, width: T) {
        self.tlwh[(0, 2)] = width;
    }

    #[inline(always)]
    pub fn height(&self) -> T {
        self.tlwh[(0, 3)]
    }

    #[inline(always)]
    pub(crate) fn set_height(&mut self, height: T) {
        self.tlwh[(0, 3)] = height;
    }

    pub fn area(&self) -> T {
        (self.tlwh[(0, 2)] + T::from(1).unwrap())
            * (self.tlwh[(0, 3)] + T::from(1).unwrap())
    }

    pub(crate) fn calc_iou(&self, other: &Rect<T>) -> T {
        let box_area = other.area();
        let iw = (self.tlwh[(0, 0)] + self.tlwh[(0, 2)])
            .min(other.tlwh[(0, 0)] + other.tlwh[(0, 2)])
            - (self.tlwh[(0, 0)]).max(other.tlwh[(0, 0)])
            + T::from(1).unwrap();

        let mut iou = T::from(0).unwrap();
        if iw > T::from(0).unwrap() {
            let ih = (self.tlwh[(0, 1)] + self.tlwh[(0, 3)])
                .min(other.tlwh[(0, 1)] + other.tlwh[(0, 3)])
                - (self.tlwh[(0, 1)]).max(other.tlwh[(0, 1)])
                + T::from(1).unwrap();

            if ih > T::from(0).unwrap() {
                let ua = (self.tlwh[(0, 2)] + T::from(1).unwrap())
                    * (self.tlwh[(0, 3)] + T::from(1).unwrap())
                    + box_area
                    - iw * ih;
                iou = iw * ih / ua;
            }
        }
        iou
    }

    pub(crate) fn get_xyah(&self) -> Xyah<T> {
        Matrix1x4::new(
            self.tlwh[(0, 0)] + self.tlwh[(0, 2)] / T::from(2).unwrap(),
            self.tlwh[(0, 1)] + self.tlwh[(0, 3)] / T::from(2).unwrap(),
            self.tlwh[(0, 2)] / self.tlwh[(0, 3)],
            self.tlwh[(0, 3)],
        )
    }
}

impl From<Rect<f32>> for PyRect {
    fn from(rect: Rect<f32>) -> Self {
        PyRect { inner: rect }
    }
}

impl From<PyRect> for Rect<f32> {
    fn from(pyrect: PyRect) -> Self {
        pyrect.inner
    }
}
