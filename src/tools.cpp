#include "tools.h"

#include <chrono>
#include <future>

double get_current_timestamp() {
    // Get the current time as the number of nanoseconds since the epoch
    const auto now = std::chrono::system_clock::now();
    const auto duration = now.time_since_epoch();

    // Convert the duration to seconds with a fractional part
    const double seconds = std::chrono::duration<double>(duration).count();

    return seconds;
}

long get_current_timestamp_micro() {
    return static_cast<long>(get_current_timestamp() * 1000 * 1000);
}

long get_current_timestamp_milli() {
    return static_cast<long>(get_current_timestamp() * 1000);
}

Eigen::MatrixXd differentiate_series(const Eigen::MatrixXd &series) {
    const int n = static_cast<int>(series.rows());
    const int m = static_cast<int>(series.cols());

    if (n < 2 || m < 2) {
        return {0, m};
    }

    const auto times = series.col(0);
    const auto values = series.rightCols(m - 1);

    // Eigen pipeline: finite differences over consecutive samples. Both the
    // time step and the value change are evaluated as expressions, so no
    // intermediate matrices are materialized until they are needed.
    const Eigen::VectorXd dt = times.tail(n - 1) - times.head(n - 1);
    const Eigen::MatrixXd delta = values.bottomRows(n - 1) - values.topRows(n - 1);

    Eigen::MatrixXd result(n - 1, m);
    result.col(0) = (times.tail(n - 1) + times.head(n - 1)) * 0.5;
    result.rightCols(m - 1) = delta.array().colwise() / dt.array();

    return result;
}

Eigen::MatrixXd integrate_series(const Eigen::MatrixXd &velocity_series,
                                 const Eigen::RowVectorXd &initial_point) {
    const int n = static_cast<int>(velocity_series.rows());
    const int m = static_cast<int>(velocity_series.cols());

    if (n < 1 || m < 2 || initial_point.size() != m) {
        return {0, m};
    }

    const auto times = velocity_series.col(0);
    const auto velocities = velocity_series.rightCols(m - 1);

    // The initial point (t, x1, x2, ...) anchors the series: it is row 0 as is.
    Eigen::MatrixXd result(n, m);
    result.row(0) = initial_point;

    if (n == 1) {
        return result;
    }

    // Eigen pipeline: trapezoidal increments for every interval at once. The
    // average velocity and the time step are expression templates, so the
    // (n - 1) x (m - 1) matrix is materialized only when needed.
    const Eigen::VectorXd dt = times.tail(n - 1) - times.head(n - 1);
    const Eigen::MatrixXd increments =
        ((velocities.topRows(n - 1) + velocities.bottomRows(n - 1)) * 0.5)
            .array().colwise() * dt.array();

    // The cumulative sum is inherently sequential, but every step is still a
    // vectorized Eigen expression over all value columns at once.
    for (int i = 1; i < n; ++i) {
        result(i, 0) = times(i);
        result.row(i).tail(m - 1) = result.row(i - 1).tail(m - 1) + increments.row(i - 1);
    }

    return result;
}
