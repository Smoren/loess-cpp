#pragma once

#include <Eigen/Dense>

class Loess {
public:
    explicit Loess(double bandwidth = 0.3, int degree = 1);

    [[nodiscard]] Eigen::MatrixXd smooth(const Eigen::MatrixXd &input_data) const;
    [[nodiscard]] Eigen::MatrixXd predict(const Eigen::MatrixXd &input_data, const Eigen::VectorXd &new_times) const;

private:
    double _bandwidth;
    int _degree;

    static double _tricube_kernel(double u);

    // Vectorized weight computation (the result is written to weights,
    // scratch is a reusable buffer for finding the k-th distance)
    void _compute_weights(const Eigen::VectorXd &t_values, double t_target,
                          Eigen::VectorXd &weights, Eigen::VectorXd &scratch) const;

    // Optimized version of the local regression (weighted least squares via moments)
    [[nodiscard]] Eigen::MatrixXd _local_regression_batch(const Eigen::VectorXd &t_values, const Eigen::MatrixXd &data, const Eigen::VectorXd &target_times) const;

    // Determinant of a small (2x2 or 3x3) matrix without memory allocation
    static double _determinant(const Eigen::MatrixXd &m);
};
