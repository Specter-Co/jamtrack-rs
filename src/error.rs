use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ByteTrackError {
    #[error("Error: {0}")]
    LapjvError(String),
    #[error("Error: {0}")]
    ExecLapjvError(String),
    #[error("Error: {0}")]
    ByteTrackerError(String),
}

impl From<ByteTrackError> for PyErr {
    fn from(err: ByteTrackError) -> PyErr {
        match err {
            ByteTrackError::LapjvError(msg) => {
                PyRuntimeError::new_err(format!("Lapjv error: {}", msg))
            }
            ByteTrackError::ExecLapjvError(msg) => {
                PyRuntimeError::new_err(format!("ExecLapjv error: {}", msg))
            }
            ByteTrackError::ByteTrackerError(msg) => {
                PyRuntimeError::new_err(format!("ByteTracker error: {}", msg))
            }
        }
    }
}
