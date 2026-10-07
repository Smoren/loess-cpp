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
    result.col(0) = input_data.col(0); // временные метки

    // Векторизованное вычисление для всех точек
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

    // Векторизованное предсказание для всех новых точек
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

Eigen::VectorXd Loess::_compute_weights(const Eigen::VectorXd& t_values, const double t_target) const {
    const int n = static_cast<int>(t_values.size());

    // Векторизованное вычисление расстояний
    Eigen::VectorXd distances = (t_values.array() - t_target).abs();

    // Находим ширину окна
    int k = std::max(static_cast<int>(_bandwidth * n), _degree + 1);
    k = std::min(k, n - 1);

    // Используем nth_element для нахождения k-го наименьшего расстояния
    Eigen::VectorXd distances_copy = distances;
    std::nth_element(distances_copy.data(), distances_copy.data() + k,
                     distances_copy.data() + distances_copy.size());
    const double max_distance = distances_copy(k);

    if (max_distance <= std::numeric_limits<double>::epsilon()) {
        return Eigen::VectorXd::Constant(n, 1.0 / n);
    }

    // Векторизованное вычисление весов
    Eigen::ArrayXd u = distances.array() / max_distance;
    Eigen::ArrayXd weights_array = (1.0 - u.pow(3)).pow(3);

    // Обнуляем веса за пределами окна
    weights_array = (u < 1.0).select(weights_array, 0.0);

    return weights_array.matrix();
}

void Loess::_build_design_matrix(const Eigen::VectorXd &t_diff, int degree, Eigen::MatrixXd &X) {
    const int n = static_cast<int>(t_diff.size());
    X.resize(n, degree + 1);

    X.col(0).setOnes(); // константный член

    // Векторизованное заполнение степеней
    for (int j = 1; j <= degree; ++j) {
        X.col(j) = t_diff.array().pow(j).matrix();
    }
}

Eigen::MatrixXd Loess::_local_regression_batch(
    const Eigen::VectorXd &t_values,
    const Eigen::MatrixXd &data,
    const Eigen::VectorXd &target_times) const {

    const int n = static_cast<int>(t_values.size());
    const int num_targets = static_cast<int>(target_times.size());
    const int num_dims = static_cast<int>(data.cols() - 1);

    Eigen::MatrixXd result(num_targets, num_dims);

    // Предварительное выделение памяти
    Eigen::MatrixXd X, XTW, XTWX;
    Eigen::VectorXd weights;
    Eigen::VectorXd t_diff(n);

    // Извлекаем матрицу Y (исключаем столбец времени)
    const Eigen::MatrixXd Y = data.rightCols(num_dims);

    for (int i = 0; i < num_targets; ++i) {
        const double t_target = target_times(i);

        // Вычисляем разности времени
        t_diff = t_values.array() - t_target;

        // Строим матрицу плана
        _build_design_matrix(t_diff, _degree, X);

        // Вычисляем веса
        weights = _compute_weights(t_values, t_target);

        // Взвешенная регрессия
        XTW = X.transpose() * weights.asDiagonal();
        XTWX = XTW * X;

        // Решаем систему для всех измерений одновременно
        Eigen::LDLT<Eigen::MatrixXd> ldlt;

        // Проверяем численную устойчивость
        if (XTWX.determinant() > 1e-10) {
            ldlt.compute(XTWX);
            if (ldlt.info() == Eigen::Success) {
                Eigen::MatrixXd beta = ldlt.solve(XTW * Y);
                result.row(i) = beta.row(0); // берем только константный член
                continue;
            }
        }

        // Fallback: используем SVD
        Eigen::JacobiSVD svd(XTWX, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd beta = svd.solve(XTW * Y);
        result.row(i) = beta.row(0);
    }

    return result;
}
