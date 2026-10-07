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

    // Векторизованная версия вычисления весов
    [[nodiscard]] Eigen::VectorXd _compute_weights(const Eigen::VectorXd &t_values, double t_target) const;

    // Оптимизированная версия локальной регрессии
    [[nodiscard]] Eigen::MatrixXd _local_regression_batch(const Eigen::VectorXd &t_values, const Eigen::MatrixXd &data, const Eigen::VectorXd &target_times) const;

    // Вспомогательный метод для построения матрицы плана
    static void _build_design_matrix(const Eigen::VectorXd &t_diff, int degree, Eigen::MatrixXd &X);
};
