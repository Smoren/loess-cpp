#include "../src/holt_winters.h"
#include "catch_amalgamated.hpp"

TEST_CASE("A constant series forecasts a constant", "[holt_winters]") {
    Eigen::MatrixXd series(6, 2);
    for (int i = 0; i < 6; ++i) {
        series.row(i) << i * 0.1, 5.0;
    }

    const HoltWinters model(1, 0.5, 0.1, 0.3);

    Eigen::VectorXd new_times(3);
    new_times << 0.6, 0.7, 0.8;

    const Eigen::MatrixXd predicted = model.predict(series, new_times);

    REQUIRE(predicted.rows() == 3);
    REQUIRE(predicted.cols() == 2);
    REQUIRE(predicted(0, 0) == Catch::Approx(0.6));

    for (int i = 0; i < predicted.rows(); ++i) {
        REQUIRE(predicted(i, 1) == Catch::Approx(5.0));
    }
}

TEST_CASE("A linear ramp with unit smoothing continues exactly", "[holt_winters]") {
    // y = 2t sampled at t = 0..3; period = 1 disables seasonality.
    Eigen::MatrixXd series(4, 2);
    series << 0.0, 0.0,
              1.0, 2.0,
              2.0, 4.0,
              3.0, 6.0;

    const HoltWinters model(1, 1.0, 1.0, 1.0);

    Eigen::VectorXd new_times(2);
    new_times << 4.0, 5.0;

    const Eigen::MatrixXd predicted = model.predict(series, new_times);

    REQUIRE(predicted(0, 1) == Catch::Approx(8.0));  // y(4)
    REQUIRE(predicted(1, 1) == Catch::Approx(10.0)); // y(5)
}

TEST_CASE("A perfectly periodic series is reproduced", "[holt_winters]") {
    // Period-2 pattern [1, 2] repeated eight times.
    Eigen::MatrixXd series(8, 2);
    for (int i = 0; i < 8; ++i) {
        series.row(i) << static_cast<double>(i), (i % 2 == 0 ? 1.0 : 2.0);
    }

    const HoltWinters model(2, 0.5, 0.0, 0.5);

    Eigen::VectorXd new_times(4);
    new_times << 8.0, 9.0, 10.0, 11.0;

    const Eigen::MatrixXd predicted = model.predict(series, new_times);

    REQUIRE(predicted(0, 1) == Catch::Approx(1.0));
    REQUIRE(predicted(1, 1) == Catch::Approx(2.0));
    REQUIRE(predicted(2, 1) == Catch::Approx(1.0));
    REQUIRE(predicted(3, 1) == Catch::Approx(2.0));
}

TEST_CASE("Multiple value columns are forecast independently", "[holt_winters]") {
    // a(t) = 2t, b(t) = 10 - 2t sampled at t = 0..5.
    Eigen::MatrixXd series(6, 3);
    for (int i = 0; i < 6; ++i) {
        series.row(i) << static_cast<double>(i), 2.0 * i, 10.0 - 2.0 * i;
    }

    const HoltWinters model(1, 1.0, 1.0, 1.0);

    Eigen::VectorXd new_times(2);
    new_times << 6.0, 7.0;

    const Eigen::MatrixXd predicted = model.predict(series, new_times);

    REQUIRE(predicted.rows() == 2);
    REQUIRE(predicted.cols() == 3);

    REQUIRE(predicted(0, 1) == Catch::Approx(12.0));
    REQUIRE(predicted(1, 1) == Catch::Approx(14.0));
    REQUIRE(predicted(0, 2) == Catch::Approx(-2.0));
    REQUIRE(predicted(1, 2) == Catch::Approx(-4.0));
}

TEST_CASE("A series shorter than one season is rejected", "[holt_winters]") {
    Eigen::MatrixXd series(3, 2);
    series << 0.0, 1.0,
              1.0, 2.0,
              2.0, 3.0;

    const HoltWinters model(5, 0.3, 0.1, 0.3);

    Eigen::VectorXd new_times(2);
    new_times << 3.0, 4.0;

    const Eigen::MatrixXd predicted = model.predict(series, new_times);

    REQUIRE(predicted.rows() == 0);
    REQUIRE(predicted.cols() == 2);
}
