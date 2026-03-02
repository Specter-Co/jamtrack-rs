use nalgebra::SMatrix;

/* -----------------------------------------------------------------------------
 * Type aliases
 * ----------------------------------------------------------------------------- */
// 1x4
pub(crate) type DetectBox = SMatrix<f32, 1, 4>;
// 1x8
pub(crate) type StateMean = SMatrix<f32, 1, 8>;
// 8x8
pub(crate) type StateCov = SMatrix<f32, 8, 8>;
// 1x4
pub(crate) type StateHMean = SMatrix<f32, 1, 4>;
// 4x4
pub(crate) type StateHCov = SMatrix<f32, 4, 4>;

/* -----------------------------------------------------------------------------
 * Kalman Filter
 * ----------------------------------------------------------------------------- */
#[derive(Debug, Clone)]
pub(crate) struct KalmanFilter {
    std_weight_position: f32,
    std_weight_velocity: f32,
    std_weight_position_meas: f32,
    std_weight_position_mot: f32,
    std_weight_velocity_mot: f32,
    std_aspect_ratio_init: f32,
    std_d_aspect_ratio_init: f32,
    std_aspect_ratio_mot: f32,
    std_d_aspect_ratio_mot: f32,
    std_aspect_ratio_meas: f32,
    update_mat: SMatrix<f32, 4, 8>, // 4x8
}

impl KalmanFilter {
    pub(crate) fn new(
        std_weight_position: f32,
        std_weight_velocity: f32,
        std_weight_position_meas: f32,
        std_weight_position_mot: f32,
        std_weight_velocity_mot: f32,
        std_aspect_ratio_init: f32,
        std_d_aspect_ratio_init: f32,
        std_aspect_ratio_mot: f32,
        std_d_aspect_ratio_mot: f32,
        std_aspect_ratio_meas: f32,
    ) -> Self {
        // Observation matrix: extracts [cx, cy, a, h] from state
        let mut update_mat = SMatrix::<f32, 4, 8>::zeros();
        update_mat[(0, 0)] = 1.0;
        update_mat[(1, 1)] = 1.0;
        update_mat[(2, 2)] = 1.0;
        update_mat[(3, 3)] = 1.0;

        Self {
            std_weight_position,
            std_weight_velocity,
            std_weight_position_meas,
            std_weight_position_mot,
            std_weight_velocity_mot,
            std_aspect_ratio_init,
            std_d_aspect_ratio_init,
            std_aspect_ratio_mot,
            std_d_aspect_ratio_mot,
            std_aspect_ratio_meas,
            update_mat,
        }
    }

    /// Build motion matrix F for given dt
    fn motion_matrix(dt: f32) -> SMatrix<f32, 8, 8> {
        let mut f = SMatrix::<f32, 8, 8>::identity();
        // Position += velocity * dt
        for i in 0..4 {
            f[(i, i + 4)] = dt;
        }
        f
    }

    pub(crate) fn initiate(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
        measurement: &DetectBox,
    ) {
        let mean_vel = SMatrix::<f32, 1, 4>::zeros();
        let mean_pos = measurement;
        mean.as_mut_slice()[0..4].copy_from_slice(mean_pos.as_slice());
        mean.as_mut_slice()[4..8].copy_from_slice(mean_vel.as_slice());

        let mut std = SMatrix::<f32, 1, 8>::zeros();
        let mesure_val = measurement[(0, 3)];
        std[0] = 2.0 * self.std_weight_position * mesure_val;
        std[1] = 2.0 * self.std_weight_position * mesure_val;
        std[2] = self.std_aspect_ratio_init;
        std[3] = 2.0 * self.std_weight_position * mesure_val;
        std[4] = 10.0 * self.std_weight_velocity * mesure_val;
        std[5] = 10.0 * self.std_weight_velocity * mesure_val;
        std[6] = self.std_d_aspect_ratio_init;
        std[7] = 10.0 * self.std_weight_velocity * mesure_val;

        let tmp = std.component_mul(&std);
        // convert 1-d array to 2-d array that has diagonal values of 1-d array
        *covariance = SMatrix::<f32, 8, 8>::from_diagonal(&tmp.transpose());
    }

    /// Predict state forward by dt seconds using continuous-discrete Kalman filter.
    ///
    /// The process noise Q is integrated properly for variable dt:
    /// - Position variance: dt³/3 * q_vel + dt * q_pos
    /// - Velocity variance: dt * q_vel
    /// - Position-velocity covariance: dt²/2 * q_vel
    pub(crate) fn predict(
        &mut self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
        dt: f32,
    ) {
        let motion_mat = Self::motion_matrix(dt);
        let h = mean[(0, 3)]; // height for scaling

        // Process noise variances (continuous-time power spectral densities)
        let q_pos = (self.std_weight_position_mot * h).powi(2);
        let q_vel = (self.std_weight_velocity_mot * h).powi(2);
        let q_a = self.std_aspect_ratio_mot.powi(2);
        let q_va = self.std_d_aspect_ratio_mot.powi(2);

        let dt2 = dt * dt;
        let dt3 = dt2 * dt;

        // Build Q matrix with proper continuous-discrete integration
        // State: [cx, cy, a, h, vcx, vcy, va, vh]
        let mut q = SMatrix::<f32, 8, 8>::zeros();

        // cx-vcx pair (indices 0, 4)
        q[(0, 0)] = dt3 / 3.0 * q_vel + dt * q_pos;
        q[(0, 4)] = dt2 / 2.0 * q_vel;
        q[(4, 0)] = dt2 / 2.0 * q_vel;
        q[(4, 4)] = dt * q_vel;

        // cy-vcy pair (indices 1, 5)
        q[(1, 1)] = dt3 / 3.0 * q_vel + dt * q_pos;
        q[(1, 5)] = dt2 / 2.0 * q_vel;
        q[(5, 1)] = dt2 / 2.0 * q_vel;
        q[(5, 5)] = dt * q_vel;

        // a-va pair (indices 2, 6) - aspect ratio
        q[(2, 2)] = dt3 / 3.0 * q_va + dt * q_a;
        q[(2, 6)] = dt2 / 2.0 * q_va;
        q[(6, 2)] = dt2 / 2.0 * q_va;
        q[(6, 6)] = dt * q_va;

        // h-vh pair (indices 3, 7)
        q[(3, 3)] = dt3 / 3.0 * q_vel + dt * q_pos;
        q[(3, 7)] = dt2 / 2.0 * q_vel;
        q[(7, 3)] = dt2 / 2.0 * q_vel;
        q[(7, 7)] = dt * q_vel;

        // Predict: x' = F*x, P' = F*P*F' + Q
        *mean = (&motion_mat * mean.transpose()).transpose();
        *covariance = motion_mat * *covariance * motion_mat.transpose() + q;
    }

    pub(crate) fn update(
        &mut self,
        mean: &mut StateMean,      // 1x8
        covariance: &mut StateCov, // 8x8
        measurement: &DetectBox,   // 1x4
    ) {
        let mut projected_mean = SMatrix::<f32, 1, 4>::zeros();
        let mut projected_covariance = SMatrix::<f32, 4, 4>::zeros();
        self.project(
            &mut projected_mean,
            &mut projected_covariance,
            &mean,
            &covariance,
        );

        let b = (*covariance * self.update_mat.transpose()).transpose();
        let choleskey_factor = projected_covariance.cholesky().unwrap();
        // kalman_gain: 8x4
        let kalman_gain = choleskey_factor.solve(&b);
        // innovation: 1x4
        let innovation = measurement - &projected_mean;
        // tmp: 1x8
        let tmp = innovation * &kalman_gain;
        *mean += &tmp;
        *covariance -=
            kalman_gain.transpose() * projected_covariance * kalman_gain;
    }

    pub(crate) fn project(
        &self,
        projected_mean: &mut StateHMean, // 1x4
        projected_covariance: &mut StateHCov, // 4x4
        mean: &StateMean,                // 1x8
        covariance: &StateCov,           // 8x8
    ) {
        let std = SMatrix::<f32, 1, 4>::from_iterator([
            self.std_weight_position_meas * mean[(0, 3)],
            self.std_weight_position_meas * mean[(0, 3)],
            self.std_aspect_ratio_meas,
            self.std_weight_position_meas * mean[(0, 3)],
        ]);

        // update_mat: 4x8, mean: 1x8
        // projected_mean: 4x1
        let tmp = mean * self.update_mat.transpose();
        *projected_mean = tmp;

        // 4x4
        let diag = SMatrix::<f32, 4, 4>::from_diagonal(&std.transpose());
        let innovation_cov = diag.component_mul(&diag);
        let cov = self.update_mat * covariance * self.update_mat.transpose();
        *projected_covariance = cov + innovation_cov;
    }
}
