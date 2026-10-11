#pragma once

#include <Eigen/Dense>

// Additive Holt-Winters / ETS(A,A,A) forecasting model.
//
// The series is decomposed into a level, a trend and a seasonal component that
// are updated recursively (exponential smoothing). The model is defined by the
// seasonal period (in samples) and the three smoothing parameters.
class HoltWinters {
public:
    // period — number of samples in one seasonal cycle (1 disables seasonality,
    // reducing the model to Holt's linear trend). The smoothing parameters are
    // clamped to [0, 1].
    explicit HoltWinters(int period = 1, double alpha = 0.3, double beta = 0.1, double gamma = 0.3);

    // Forecasts the series at new_times. The model is fitted on input_data on
    // every call; the horizon is measured in samples relative to the last
    // observed sample, so times beyond the last sample are extrapolated.
    [[nodiscard]] Eigen::MatrixXd predict(const Eigen::MatrixXd &input_data, const Eigen::VectorXd &new_times) const;

private:
    int _period;
    double _alpha;
    double _beta;
    double _gamma;
};
