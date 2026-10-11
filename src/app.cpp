#include "app.h"

Eigen::MatrixXd generate_custom_time_series() {
    TimeSeriesParams params = {
        .t_max = 10.0,
        .step = 0.1,
        .sine_amplitude = 0.3,
        .sine_frequency = 0.5,
        .asinh_amplitude = 0.5,
        .asinh_scale = 2.0,
        .intercept = 0.0,
        .noise_stddev = 0.3,
    };

    return generate_time_series(params);
}

Loess create_loess_quadratic() {
    return Loess(0.5, 2);
}

Loess create_loess_linear() {
    return Loess(0.5, 1);
}

void print_time_series(const Eigen::MatrixXd& series) {
    for (int i = 0; i < series.rows(); ++i) {
        std::cout << "(";
        for (int j = 0; j < series.cols(); ++j) {
            if (j > 0) {
                std::cout << ";";
            }
            std::cout << series(i, j);
        }
        std::cout << ")";
    }
    std::cout << "\n";
}

void test_loess() {
    const auto loess_quadratic = create_loess_quadratic();
    const auto loess_linear = create_loess_linear();

    const auto series = generate_custom_time_series();
    std::cout << "Generated points: " << series.rows() << std::endl;

    const Eigen::MatrixXd smoothed = loess_linear.smooth(series);
    std::cout << "Smoothed points: " << smoothed.rows() << std::endl;

    // Prediction at new time points
    Eigen::VectorXd new_times = series.col(0).array() + series.row(series.rows()-1).col(0).value();
    const Eigen::MatrixXd predicted = loess_quadratic.predict(smoothed, new_times);
    std::cout << "Predicted points: " << predicted.rows() << std::endl << std::endl;

    const auto velocities = differentiate_series(series);
    const auto velocities_smoothed = loess_linear.smooth(velocities);
    const auto velocities_predicted = loess_linear.predict(velocities_smoothed, new_times);
    const auto predicted_by_velocities = integrate_series(velocities_predicted, smoothed.row(series.rows()-1));

    std::cout << "series:" << std::endl;
    print_time_series(series);
    std::cout << std::endl << "smoothed:" << std::endl;
    print_time_series(smoothed);
    std::cout << std::endl << "predicted:" << std::endl;
    print_time_series(predicted);
    std::cout << std::endl << "predicted by velocities:" << std::endl;
    print_time_series(predicted_by_velocities);
    std::cout << std::endl << "velocities smoothed:" << std::endl;
    print_time_series(velocities_smoothed);
    std::cout << std::endl << "velocities predicted:" << std::endl;
    print_time_series(velocities_predicted);
    std::cout << std::endl;
}

void run_app() {
    test_loess();

    const auto series = generate_custom_time_series();
    const auto loess = create_loess_quadratic();

    // Prediction at new time points
    Eigen::VectorXd new_times = series.col(0).array() + series.row(series.rows()-1).col(0).value();

    benchmark("Smoothing benchmark", 100, [&] { return loess.smooth(series); });
    benchmark("Prediction benchmark", 100, [&] { return loess.predict(series, new_times); });
}
