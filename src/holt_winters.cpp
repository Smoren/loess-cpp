#include "holt_winters.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

HoltWinters::HoltWinters(const int period, const double alpha, const double beta, const double gamma)
    : _period(std::max(1, period))
    , _alpha(std::clamp(alpha, 0.0, 1.0))
    , _beta(std::clamp(beta, 0.0, 1.0))
    , _gamma(std::clamp(gamma, 0.0, 1.0)) {}

Eigen::MatrixXd HoltWinters::predict(const Eigen::MatrixXd& input_data, const Eigen::VectorXd& new_times) const {
    assert(input_data.cols() >= 2);
    const int n = static_cast<int>(input_data.rows());
    const int num_dims = static_cast<int>(input_data.cols() - 1);

    if (n < _period + 1) {
        return {0, input_data.cols()};
    }

    const Eigen::MatrixXd Y = input_data.rightCols(num_dims);

    // Seasonal indices via classical decomposition: detrend with a centered
    // moving average and average the detrended values per phase. This is much
    // more robust than using the raw first season, which is contaminated by the
    // trend (e.g. a steeply rising trend makes the first season look seasonal).
    const int half = _period / 2;
    Eigen::MatrixXd seasonal = Eigen::MatrixXd::Zero(_period, num_dims);
    {
        // The centered moving average is defined on the interior; outside it the
        // trend is held constant to avoid the bias of a one-sided window.
        const int first = half;
        const int last = n - 1 - half;
        Eigen::MatrixXd trend_ma(n, num_dims);
        for (int i = first; i <= last; ++i) {
            Eigen::RowVectorXd acc = Eigen::RowVectorXd::Zero(num_dims);
            double sum_w = 0.0;
            for (int k = i - half; k <= i + half; ++k) {
                const double w = (k == i - half || k == i + half) ? 0.5 : 1.0;
                acc += w * Y.row(k);
                sum_w += w;
            }
            trend_ma.row(i) = acc / sum_w;
        }
        for (int i = 0; i < first; ++i) {
            trend_ma.row(i) = trend_ma.row(first);
        }
        for (int i = last + 1; i < n; ++i) {
            trend_ma.row(i) = trend_ma.row(last);
        }

        Eigen::MatrixXd seasonal_sum = Eigen::MatrixXd::Zero(_period, num_dims);
        std::vector<int> counts(_period, 0);
        for (int i = 0; i < n; ++i) {
            seasonal_sum.row(i % _period) += Y.row(i) - trend_ma.row(i);
            counts[i % _period]++;
        }
        for (int j = 0; j < _period; ++j) {
            if (counts[j] > 0) {
                seasonal.row(j) = seasonal_sum.row(j) / static_cast<double>(counts[j]);
            }
        }
        // Center the seasonal component: its mean over a period is zero.
        seasonal.rowwise() -= seasonal.colwise().mean();
    }

    // Initialize the level and trend from the deseasonalized series: the level is
    // the mean of the first season, the trend is the average per-sample change
    // between the first two seasons (zero if there is only one season).
    Eigen::MatrixXd deseasonalized(n, num_dims);
    for (int i = 0; i < n; ++i) {
        deseasonalized.row(i) = Y.row(i) - seasonal.row(i % _period);
    }

    Eigen::RowVectorXd level = deseasonalized.topRows(_period).colwise().mean();
    Eigen::RowVectorXd trend = Eigen::RowVectorXd::Zero(num_dims);
    if (n >= 2 * _period) {
        const Eigen::RowVectorXd first = deseasonalized.topRows(_period).colwise().mean();
        const Eigen::RowVectorXd second = deseasonalized.middleRows(_period, _period).colwise().mean();
        trend = (second - first) / static_cast<double>(_period);
    }

    // Recursive update over the observed samples (vectorized over the value columns).
    for (int t = _period; t < n; ++t) {
        const int j = t % _period;
        const Eigen::RowVectorXd level_old = level;
        const Eigen::RowVectorXd trend_old = trend;

        level = _alpha * (Y.row(t) - seasonal.row(j)) + (1.0 - _alpha) * (level_old + trend_old);
        trend = _beta * (level - level_old) + (1.0 - _beta) * trend_old;
        seasonal.row(j) = _gamma * (Y.row(t) - level_old - trend_old) + (1.0 - _gamma) * seasonal.row(j);
    }

    // Forecast at each requested time. The horizon h is the number of samples
    // between the requested time and the last observed sample.
    const double step = n >= 2 ? input_data(1, 0) - input_data(0, 0) : 1.0;
    const double t_last = input_data(n - 1, 0);

    const int m = static_cast<int>(new_times.size());
    Eigen::MatrixXd result(m, input_data.cols());
    result.col(0) = new_times;

    for (int i = 0; i < m; ++i) {
        const int h = static_cast<int>(std::llround((new_times(i) - t_last) / step));
        const int idx = ((n - 1 + h) % _period + _period) % _period;
        result.row(i).tail(num_dims) = level + static_cast<double>(h) * trend + seasonal.row(idx);
    }

    return result;
}
