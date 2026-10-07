#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "loess.h"

Loess::Loess(const double bandwidth, const int degree)
    : _bandwidth(std::max(0.05, std::min(1.0, bandwidth)))
    , _degree(std::max(1, std::min(2, degree))) {}

Eigen::MatrixXd Loess::smooth(const Eigen::MatrixXd& input_data) const {
    assert(input_data.cols() >= 2);
    const int n = static_cast<int>(input_data.rows());

    if (n < _degree + 1) {
        return {0, input_data.cols()};
    }

    Eigen::MatrixXd result(n, input_data.cols());
    result.col(0) = input_data.col(0); // time stamps

    // Vectorized computation for all points
    Eigen::MatrixXd smoothed = _local_regression_batch(input_data.col(0), input_data, input_data.col(0));

    result.rightCols(input_data.cols() - 1) = smoothed;

    return result;
}

Eigen::MatrixXd Loess::predict(const Eigen::MatrixXd& input_data, const Eigen::VectorXd& new_times) const {
    assert(input_data.cols() >= 2);
    const int n = static_cast<int>(new_times.size());
    const int input_n = static_cast<int>(input_data.rows());

    if (input_n < _degree + 1) {
        return {0, input_data.cols()};
    }

    Eigen::MatrixXd result(n, input_data.cols());
    result.col(0) = new_times;

    // Vectorized prediction for all new points
    Eigen::MatrixXd predicted = _local_regression_batch(input_data.col(0), input_data, new_times);

    result.rightCols(input_data.cols() - 1) = predicted;

    return result;
}

double Loess::_tricube_kernel(const double u) {
    if (std::abs(u) >= 1.0) {
        return 0.0;
    }
    const double v = 1.0 - std::pow(std::abs(u), 3);
    return v * v * v;
}

void Loess::_compute_weights(const Eigen::VectorXd& t_values, const double t_target,
                             Eigen::VectorXd& weights, Eigen::VectorXd& scratch) const {
    const int n = static_cast<int>(t_values.size());

    // Distances are stored temporarily in weights (buffer reuse, no allocations)
    weights = (t_values.array() - t_target).abs();

    // Determine the window width
    int k = std::max(static_cast<int>(_bandwidth * n), _degree + 1);
    k = std::min(k, n - 1);

    // Find the k-th smallest distance in a separate buffer so distances stay intact
    scratch = weights;
    std::nth_element(scratch.data(), scratch.data() + k,
                     scratch.data() + scratch.size());
    const double max_distance = scratch(k);

    if (max_distance <= std::numeric_limits<double>::epsilon()) {
        weights.setConstant(1.0 / n);
        return;
    }

    // Tricube kernel weights: w = (1 - u^3)^3 for u < 1, otherwise 0.
    // Manual loop without temporary arrays and without calling std::pow.
    const double inv_max = 1.0 / max_distance;
    for (int i = 0; i < n; ++i) {
        const double u = weights(i) * inv_max;
        if (u < 1.0) {
            const double v = 1.0 - u * u * u;
            weights(i) = v * v * v;
        } else {
            weights(i) = 0.0;
        }
    }
}

double Loess::_determinant(const Eigen::MatrixXd &m) {
    const int d = static_cast<int>(m.rows());
    if (d == 1) {
        return m(0, 0);
    }
    if (d == 2) {
        return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    }
    if (d == 3) {
        return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
             - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
             + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    }
    return m.determinant();
}

Eigen::MatrixXd Loess::_local_regression_batch(
    const Eigen::VectorXd &t_values,
    const Eigen::MatrixXd &data,
    const Eigen::VectorXd &target_times) const {

    const int n = static_cast<int>(t_values.size());
    const int num_targets = static_cast<int>(target_times.size());
    const int num_dims = static_cast<int>(data.cols() - 1);
    const int d = _degree + 1;

    Eigen::MatrixXd result(num_targets, num_dims);

    // Extract the Y matrix (excluding the time column)
    const Eigen::MatrixXd Y = data.rightCols(num_dims);

    // All buffers are allocated once and reused across the loop
    Eigen::VectorXd t_diff(n), weights(n), scratch(n), wdt(n);
    Eigen::MatrixXd WY(n, num_dims), XTWX(d, d), RHS(d, num_dims);
    Eigen::VectorXd moments(2 * _degree + 1);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(d);

    for (int i = 0; i < num_targets; ++i) {
        const double t_target = target_times(i);

        // Compute time differences
        t_diff = t_values.array() - t_target;

        // Compute weights
        _compute_weights(t_values, t_target, weights, scratch);

        // Moments m_p = sum_i w_i * dt_i^p. The matrix X^T W X is a Hankel matrix:
        // (X^T W X)(j,k) = m_{j+k}, and (X^T W Y)(j,m) = sum_i w_i dt_i^j Y(i,m).
        // This is noticeably faster than GEMM with the small dimension d.
        wdt = weights;
        for (int p = 0; p <= 2 * _degree; ++p) {
            moments(p) = wdt.sum();
            if (p < 2 * _degree) {
                wdt.array() *= t_diff.array();
            }
        }
        wdt = weights;
        for (int j = 0; j <= _degree; ++j) {
            WY = Y.array().colwise() * wdt.array();
            RHS.row(j) = WY.colwise().sum();
            wdt.array() *= t_diff.array();
        }
        for (int j = 0; j <= _degree; ++j) {
            for (int k = j; k <= _degree; ++k) {
                XTWX(j, k) = moments(j + k);
                XTWX(k, j) = moments(j + k);
            }
        }

        // Check numerical stability without allocations for the LU factorization
        if (_determinant(XTWX) > 1e-10) {
            ldlt.compute(XTWX);
            if (ldlt.info() == Eigen::Success) {
                ldlt.solveInPlace(RHS); // solve in place, without a temporary matrix
                result.row(i) = RHS.row(0); // keep only the constant term
                continue;
            }
        }

        // Fallback: use SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(XTWX, Eigen::ComputeThinU | Eigen::ComputeThinV);
        result.row(i) = svd.solve(RHS).row(0);
    }

    return result;
}
