#include "generator.h"

#include <cmath>
#include <numbers>
#include <random>

Eigen::MatrixXd generate_time_series(const TimeSeriesParams &params) {
    const double step = params.step > 0.0 ? params.step : 1.0;
    const double t_max = std::max(0.0, params.t_max);

    // Number of points: t from 0 to t_max inclusive with step
    const int n = static_cast<int>(std::floor(t_max / step)) + 1;

    Eigen::MatrixXd result(n, 2);

    // Time stamps: t_i = i * step
    result.col(0) = Eigen::VectorXd::LinSpaced(n, 0.0, static_cast<double>(n - 1) * step);

    const double omega = 2.0 * std::numbers::pi * params.sine_frequency;

    std::mt19937 rng(params.seed);
    const bool add_noise = params.noise_stddev > 0.0;
    std::normal_distribution<double> noise(0.0, add_noise ? params.noise_stddev : 1.0);

    for (int i = 0; i < n; ++i) {
        const double t = result(i, 0);

        double x = params.intercept
                 + params.sine_amplitude * std::sin(omega * t + params.sine_phase)
                 + params.asinh_amplitude * std::asinh(params.asinh_scale * t);

        if (add_noise) {
            x += noise(rng);
        }

        result(i, 1) = x;
    }

    return result;
}
