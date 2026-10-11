#include "../src/tools.h"
#include "catch_amalgamated.hpp"

TEST_CASE("Velocities of a linear ramp are constant", "[tools][velocity]") {
    Eigen::MatrixXd series(4, 2);
    series << 0.0, 0.0,
              1.0, 2.0,
              2.0, 4.0,
              3.0, 6.0;

    const Eigen::MatrixXd velocity = differentiate_series(series);

    REQUIRE(velocity.rows() == 3);
    REQUIRE(velocity.cols() == 2);

    // Time stamps are the interval midpoints.
    REQUIRE(velocity(0, 0) == Catch::Approx(0.5));
    REQUIRE(velocity(1, 0) == Catch::Approx(1.5));
    REQUIRE(velocity(2, 0) == Catch::Approx(2.5));

    // The slope is 2 everywhere.
    for (int i = 0; i < velocity.rows(); ++i) {
        REQUIRE(velocity(i, 1) == Catch::Approx(2.0));
    }
}

TEST_CASE("Velocities handle multiple indicators and uneven spacing", "[tools][velocity]") {
    Eigen::MatrixXd series(3, 3);
    series << 0.0, 0.0, 0.0,
              1.0, 3.0, -1.0,
              3.0, 7.0, 1.0;

    const Eigen::MatrixXd velocity = differentiate_series(series);

    REQUIRE(velocity.rows() == 2);
    REQUIRE(velocity.cols() == 3);

    REQUIRE(velocity(0, 0) == Catch::Approx(0.5));
    REQUIRE(velocity(0, 1) == Catch::Approx(3.0)); // (3 - 0) / 1
    REQUIRE(velocity(0, 2) == Catch::Approx(-1.0)); // (-1 - 0) / 1

    REQUIRE(velocity(1, 0) == Catch::Approx(2.0));
    REQUIRE(velocity(1, 1) == Catch::Approx(2.0)); // (7 - 3) / 2
    REQUIRE(velocity(1, 2) == Catch::Approx(1.0)); // (1 - (-1)) / 2
}

TEST_CASE("Velocities of a too-short series are empty", "[tools][velocity]") {
    Eigen::MatrixXd series(1, 2);
    series << 0.0, 5.0;

    const Eigen::MatrixXd velocity = differentiate_series(series);

    REQUIRE(velocity.rows() == 0);
    REQUIRE(velocity.cols() == 2);
}

TEST_CASE("Integration of a constant velocity is linear", "[tools][integrate]") {
    Eigen::MatrixXd velocity(3, 2);
    velocity << 0.5, 2.0,
                1.5, 2.0,
                2.5, 2.0;

    // The initial point (t, x) anchors the series.
    Eigen::RowVectorXd initial(2);
    initial << 0.5, 0.0;

    const Eigen::MatrixXd position = integrate_series(velocity, initial);

    REQUIRE(position.rows() == 3);
    REQUIRE(position.cols() == 2);

    // Row 0 is the initial point; the remaining time stamps come from the input.
    REQUIRE(position(0, 0) == Catch::Approx(0.5));
    REQUIRE(position(1, 0) == Catch::Approx(1.5));
    REQUIRE(position(2, 0) == Catch::Approx(2.5));

    // Constant velocity 2 with unit steps gives 0, 2, 4.
    REQUIRE(position(0, 1) == Catch::Approx(0.0));
    REQUIRE(position(1, 1) == Catch::Approx(2.0));
    REQUIRE(position(2, 1) == Catch::Approx(4.0));
}

TEST_CASE("Integration handles multiple indicators and uneven spacing", "[tools][integrate]") {
    Eigen::MatrixXd velocity(3, 3);
    velocity << 0.0, 3.0, -1.0,
                1.0, 2.0, 1.0,
                3.0, 5.0, 0.0;

    Eigen::RowVectorXd initial(3);
    initial << 0.0, 0.0, 0.0;

    const Eigen::MatrixXd position = integrate_series(velocity, initial);

    REQUIRE(position.rows() == 3);
    REQUIRE(position.cols() == 3);

    // Column 1: 0, then +((3+2)/2)*1 = 2.5, then +((2+5)/2)*2 = 9.5
    REQUIRE(position(0, 1) == Catch::Approx(0.0));
    REQUIRE(position(1, 1) == Catch::Approx(2.5));
    REQUIRE(position(2, 1) == Catch::Approx(9.5));

    // Column 2: 0, then +((-1+1)/2)*1 = 0, then +((1+0)/2)*2 = 1
    REQUIRE(position(0, 2) == Catch::Approx(0.0));
    REQUIRE(position(1, 2) == Catch::Approx(0.0));
    REQUIRE(position(2, 2) == Catch::Approx(1.0));
}

TEST_CASE("Differentiation then integration recovers a linear series", "[tools][integrate]") {
    // A linear position series x(t) = 1 + 2t sampled at t = 0..3.
    Eigen::MatrixXd series(4, 2);
    series << 0.0, 1.0,
              1.0, 3.0,
              2.0, 5.0,
              3.0, 7.0;

    const Eigen::MatrixXd velocity = differentiate_series(series);

    // The initial point is the linear function at the first midpoint (t = 0.5).
    Eigen::RowVectorXd initial(2);
    initial << 0.5, 1.0 + 2.0 * 0.5;

    const Eigen::MatrixXd position = integrate_series(velocity, initial);

    // The recovered samples lie on the same line: x(t) = 1 + 2t.
    REQUIRE(position.rows() == 3);
    for (int i = 0; i < position.rows(); ++i) {
        const double t = position(i, 0);
        REQUIRE(position(i, 1) == Catch::Approx(1.0 + 2.0 * t));
    }
}

TEST_CASE("Integration of a single row returns the initial point", "[tools][integrate]") {
    Eigen::MatrixXd velocity(1, 2);
    velocity << 4.0, 1.0;

    Eigen::RowVectorXd initial(2);
    initial << 4.0, 9.0;

    const Eigen::MatrixXd position = integrate_series(velocity, initial);

    REQUIRE(position.rows() == 1);
    REQUIRE(position(0, 0) == Catch::Approx(4.0));
    REQUIRE(position(0, 1) == Catch::Approx(9.0));
}

TEST_CASE("Integration accepts a series row as the initial point", "[tools][integrate]") {
    Eigen::MatrixXd series(3, 2);
    series << 0.0, 1.0,
              1.0, 3.0,
              2.0, 5.0;

    const Eigen::MatrixXd velocity = differentiate_series(series);

    // A row of a series is exactly the expected (t, x1, x2, ...) shape.
    const Eigen::MatrixXd position = integrate_series(velocity, series.row(0));

    REQUIRE(position.rows() == 2);
    REQUIRE(position(0, 0) == Catch::Approx(0.0));
    REQUIRE(position(0, 1) == Catch::Approx(1.0));
}

TEST_CASE("Integration rejects a mismatched initial point", "[tools][integrate]") {
    Eigen::MatrixXd velocity(3, 3);
    velocity << 0.0, 1.0, 2.0,
                1.0, 1.0, 2.0,
                2.0, 1.0, 2.0;

    // Only two elements while the series has three columns.
    Eigen::RowVectorXd initial(2);
    initial << 0.0, 0.0;

    const Eigen::MatrixXd position = integrate_series(velocity, initial);

    REQUIRE(position.rows() == 0);
    REQUIRE(position.cols() == 3);
}
