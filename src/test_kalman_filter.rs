/* -----------------------------------------------------------------------------
 * Tests
 * -------------------------------------------------------------------------------*/
use super::kalman_filter::KalmanFilter;
use nalgebra::{self, SMatrix};
use nearly_eq::assert_nearly_eq;

#[test]
fn test_initiate() {
    let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160., 1.0 / 20., 1.0 / 20., 1.0 / 160., 1e-2, 1e-5, 1e-2, 1e-5, 1e-1);
    let mut mean = SMatrix::<f32, 1, 8>::zeros();
    let mut covariance = SMatrix::<f32, 8, 8>::zeros();
    let measurement =
        SMatrix::<f32, 1, 4>::from_iterator(vec![1.0, 2.0, 3.0, 4.0]);

    kalman_filter.initiate(&mut mean, &mut covariance, &measurement);

    // Assert the values of mean and covariance after initiation
    let expected = SMatrix::<f32, 1, 8>::from_iterator(vec![
        1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    assert_eq!(mean, expected);
    #[rustfmt::skip]
    let expected = SMatrix::<f32, 8, 8>::from_iterator([
        0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 6.25e-2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 6.25e-2, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-10, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.25e-2,
    ]);
    for (i, &v) in covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_predict() {
    let mut kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160., 1.0 / 20., 1.0 / 20., 1.0 / 160., 1e-2, 1e-5, 1e-2, 1e-5, 1e-1);
    let mut mean = SMatrix::<f32, 1, 8>::from_iterator([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]);
    #[rustfmt::skip]
    let mut covariance = SMatrix::<f32, 8, 8>::from_iterator([
        0.2, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,      0.0,
        0.0, 0.2, 0.0,  0.0, 0.0, 0.0, 0.0,      0.0,
        0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0,      0.0,
        0.0, 0.0, 0.0,  0.2, 0.0, 0.0, 0.0,      0.0,
        0.0, 0.0, 0.0,  0.0, 4.0, 0.0, 0.0,      0.0,
        0.0, 0.0, 0.0,  0.0, 0.0, 4.0, 0.0,      0.0,
        0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.000001, 0.0,
        0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,      4.0,
    ]);

    kalman_filter.predict(&mut mean, &mut covariance, 0.1);

    // Assert the values of mean after prediction (with dt=0.1)
    // mean' = F * mean where F has dt=0.1 in off-diagonal: pos += vel * 0.1
    assert_eq!(
        mean,
        SMatrix::<f32, 1, 8>::from_iterator([
            1.5, 2.6, 3.7, 4.8, 5.0, 6.0, 7.0, 8.0
        ])
    );

    // Expected covariance with continuous-discrete Q matrix (dt=0.1, h=4)
    // P' = F*P*F' + Q where Q has position-velocity cross-covariance terms
    // q_pos = (1/20 * 4)^2 = 0.04, q_vel = (1/160 * 4)^2 = 0.000625
    // Q[i,i] (pos) = dt^3/3 * q_vel + dt * q_pos ≈ 0.004
    // Q[i,j] (pos-vel) = dt^2/2 * q_vel ≈ 3.125e-6
    // Q[j,j] (vel) = dt * q_vel ≈ 6.25e-5
    // F*P*F'[0,0] = P[0,0] + 2*dt*P[0,4] + dt^2*P[4,4] = 0.2 + 0 + 0.04 = 0.24
    // F*P*F'[0,4] = P[0,4] + dt*P[4,4] = 0 + 0.4 = 0.4
    #[rustfmt::skip]
    let expected = SMatrix::<f32, 8, 8>::from_iterator([
        // P' = F*P*F' + Q for each position-velocity pair
        // cx row: [0.24 + 0.004, 0, 0, 0, 0.4 + 3.125e-6, 0, 0, 0]
        0.244,       0.0,         0.0,         0.0,         0.400003125, 0.0,         0.0,          0.0,
        0.0,         0.244,       0.0,         0.0,         0.0,         0.400003125, 0.0,          0.0,
        0.0,         0.0,         1.01e-2,     0.0,         0.0,         0.0,         1.0e-6,       0.0,
        0.0,         0.0,         0.0,         0.244,       0.0,         0.0,         0.0,          0.400003125,
        0.400003125, 0.0,         0.0,         0.0,         4.0000625,   0.0,         0.0,          0.0,
        0.0,         0.400003125, 0.0,         0.0,         0.0,         4.0000625,   0.0,          0.0,
        0.0,         0.0,         1.0e-6,      0.0,         0.0,         0.0,         1.0e-6,       0.0,
        0.0,         0.0,         0.0,         0.400003125, 0.0,         0.0,         0.0,          4.0000625,
    ]);
    for (i, &v) in covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_project() {
    let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160., 1.0 / 20., 1.0 / 20., 1.0 / 160., 1e-2, 1e-5, 1e-2, 1e-5, 1e-1);
    let mean = SMatrix::<f32, 1, 8>::from_iterator([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]);
    #[rustfmt::skip]
    let covariance = SMatrix::<f32, 8, 8>::from_iterator([
        4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0      ,
        0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0      ,
        0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0,
        0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0      ,
        4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0  ,
        0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0  ,
        0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0 ,
        0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625  ,
    ]);
    let mut projected_mean = SMatrix::<f32, 1, 4>::zeros();
    let mut projected_covariance = SMatrix::<f32, 4, 4>::zeros();

    kalman_filter.project(
        &mut projected_mean,
        &mut projected_covariance,
        &mean,
        &covariance,
    );

    assert_eq!(
        projected_mean,
        SMatrix::<f32, 1, 4>::from_iterator([1., 2., 3., 4.])
    );
    #[rustfmt::skip]
    let expected = SMatrix::<f32, 4, 4>::from_iterator([
        4.28,   0.,     0.,     0.    ,
        0.,     4.28,   0.,     0.    ,
        0.,     0.,     0.0201, 0.    ,
        0.,     0.,     0.,     4.28  ]);
    for (i, &v) in projected_covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_update() {
    let mut kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160., 1.0 / 20., 1.0 / 20., 1.0 / 160., 1e-2, 1e-5, 1e-2, 1e-5, 1e-1);
    let mut mean = SMatrix::<f32, 1, 8>::from_iterator([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]);
    #[rustfmt::skip]
    let mut covariance = SMatrix::<f32, 8, 8>::from_iterator([
        4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0      ,
        0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0      ,
        0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0,
        0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0      ,
        4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0  ,
        0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0  ,
        0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0 ,
        0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625  ,
    ]);

    let measurement = SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
    kalman_filter.update(&mut mean, &mut covariance, &measurement);

    // Assert the values of mean and covariance after update
    assert_eq!(
        mean,
        SMatrix::<f32, 1, 8>::from_iterator([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        ])
    );
    #[rustfmt::skip]
    let expected = SMatrix::<f32, 8, 8>::from_iterator([
       3.96261682e-02, 0.0, 0.0, 0.0,3.73831776e-02, 0.0, 0.0, 0.0 ,
       0.0, 3.96261682e-02, 0.0, 0.0, 0.0, 3.73831776e-02, 0.0, 0.0,
       0.0, 0.0, 5.02487562e-03, 0.0, 0.0, 0.0, 4.97512438e-07, 0.0,
       0.0, 0.0, 0.0, 3.96261682e-02, 0.0, 0.0, 0.0, 3.73831776e-02,
       3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01, 0.0, 0.0, 0.0,
       0.0, 3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01, 0.0, 0.0,
       0.0, 0.0, 4.97512438e-07, 0.0, 0.0, 0.0, 9.99950249e-07, 0.0,
       0.0, 0.0, 0.0, 3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01]);
    for (i, &v) in covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_complex_predict() {
    let mut kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160., 1.0 / 20., 1.0 / 20., 1.0 / 160., 1e-2, 1e-5, 1e-2, 1e-5, 1e-1);
    // With constant measurement updates, mean should converge to measurement position with zero velocity
    let expected_mean = SMatrix::<f32, 1, 8>::from_iterator([
        1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    // Expected covariance after 10 update-predict cycles with dt=0.1
    // Computed with continuous-discrete Q matrix integration
    #[rustfmt::skip]
    let expected_covariance = SMatrix::<f32, 8, 8>::from_iterator([
        2.6337482e-2, 0.0,          0.0,          0.0,          1.6851421e-2, 0.0,          0.0,          0.0,
        0.0,          2.6337482e-2, 0.0,          0.0,          0.0,          1.6851421e-2, 0.0,          0.0,
        0.0,          0.0,          1.9065372e-4, 0.0,          0.0,          0.0,          1.6179691e-10, 0.0,
        0.0,          0.0,          0.0,          2.6337482e-2, 0.0,          0.0,          0.0,          1.6851421e-2,
        1.6851421e-2, 0.0,          0.0,          0.0,          3.801089e-2,  0.0,          0.0,          0.0,
        0.0,          1.6851421e-2, 0.0,          0.0,          0.0,          3.801089e-2,  0.0,          0.0,
        0.0,          0.0,          1.6179691e-10, 0.0,          0.0,          0.0,          2.1e-10,      0.0,
        0.0,          0.0,          0.0,          1.6851421e-2, 0.0,          0.0,          0.0,          3.801089e-2,
    ]);

    let mut mean = SMatrix::<f32, 1, 8>::zeros();
    let mut covariance = SMatrix::<f32, 8, 8>::zeros();
    let measurement = SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
    kalman_filter.initiate(&mut mean, &mut covariance, &measurement);

    for _ in 0..10 {
        kalman_filter.update(&mut mean, &mut covariance, &measurement);
        kalman_filter.predict(&mut mean, &mut covariance, 0.1);
    }
    kalman_filter.predict(&mut mean, &mut covariance, 0.1);

    assert_eq!(mean, expected_mean);
    for (i, (&actual, &expected)) in covariance.iter().zip(expected_covariance.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "covariance mismatch at index {}: actual={:e}, expected={:e}",
            i, actual, expected
        );
    }
}
