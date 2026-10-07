#pragma once

#include <Eigen/Dense>

// Parameters for generating a synthetic time series.
struct TimeSeriesParams {
    double t_max = 10.0;          // end time: t goes from 0 to t_max
    double step = 0.01;           // time step

    double sine_amplitude = 1.0;  // sine coefficient
    double sine_frequency = 1.0;  // sine frequency
    double sine_phase = 0.0;      // sine phase offset

    double asinh_amplitude = 1.0; // inverse hyperbolic sine coefficient
    double asinh_scale = 1.0;     // scale of the asinh argument

    double intercept = 2.0;       // free term
    double noise_stddev = 1.0;    // standard deviation of the additive Gaussian noise
    unsigned int seed = 42;       // seed for the noise generator
};

// Generates a time series as a matrix with two columns (t, x):
//   t = 0, step, 2*step, ..., t_max
//   x(t) = intercept
//        + sine_amplitude  * sin(2*pi*sine_frequency*t + sine_phase)
//        + asinh_amplitude * asinh(asinh_scale * t)
//        + noise
// The number of rows equals floor(t_max / step) + 1.
[[nodiscard]] Eigen::MatrixXd generate_time_series(const TimeSeriesParams &params);
