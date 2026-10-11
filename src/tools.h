#pragma once

#include <Eigen/Dense>

double get_current_timestamp();
long get_current_timestamp_micro();
long get_current_timestamp_milli();

// Extracts the rate-of-change (velocity) series from a time series.
//
// The input is expected to have at least two columns: column 0 holds the time
// stamps (strictly increasing), the remaining columns hold the observed values.
// For every pair of consecutive samples the function computes
//
//     v_i = (x_{i+1} - x_i) / (t_{i+1} - t_i)
//
// and returns an (n - 1) x m matrix whose first column holds the interval
// midpoints (t_i + t_{i+1}) / 2 and whose remaining columns hold the
// corresponding velocities.
//
// A series with fewer than two rows (or without value columns) yields an empty
// 0 x m matrix.
[[nodiscard]] Eigen::MatrixXd differentiate_series(const Eigen::MatrixXd &series);

// Integrates a rate-of-change (velocity) series back into a value (position)
// series, anchored at an initial point, using the trapezoidal rule.
//
// The velocity series is expected to have at least two columns: column 0 holds
// the time stamps (strictly increasing), the remaining columns hold the
// velocities. initial_point is the first point of the resulting series: its
// first element is the time stamp and the remaining elements are the values of
// the indicators at that time, so its size must equal the number of columns of
// the velocity series (m). It is naturally a row of a series, i.e.
// `series.row(i)`.
//
// The result has the same shape as the velocity series. Row 0 is initial_point,
// and each following row is the cumulative integral of the corresponding
// velocity column, using the time stamps of the velocity series:
//
//     x_0 = initial_point values
//     x_i = x_{i-1} + (v_{i-1} + v_i) / 2 * (t_i - t_{i-1})
//
// A series with fewer than one row, without value columns, or an initial_point
// of the wrong size yields an empty 0 x m matrix.
[[nodiscard]] Eigen::MatrixXd integrate_series(const Eigen::MatrixXd &velocity_series,
                                               const Eigen::RowVectorXd &initial_point);
