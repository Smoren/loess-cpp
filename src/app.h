#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "generator.h"
#include "loess.h"
#include "tools.h"

// Generates the demo time series used by the application.
[[nodiscard]] Eigen::MatrixXd generate_custom_time_series();

// Creates the demo LOESS model.
[[nodiscard]] Loess create_loess();

// Prints a time series in the format (t1;x1)(t2;x2)...
// The first column is treated as time, the rest as values.
void print_time_series(const Eigen::MatrixXd& series);

// Benchmarks an operation a given number of times
// and prints the mean and maximum execution time.
template <typename Operation>
void benchmark(const std::string& label, const int repetitions, Operation operation) {
    std::vector<long> durations_us;
    durations_us.reserve(repetitions);

    double checksum = 0.0;
    for (int i = 0; i < repetitions; ++i) {
        const long start = get_current_timestamp_micro();
        const Eigen::MatrixXd result = operation();
        const long finish = get_current_timestamp_micro();

        durations_us.push_back(finish - start);
        checksum += result.sum(); // use the result so the call is not optimized away
    }

    const long total_us = std::accumulate(durations_us.begin(), durations_us.end(), 0L);
    const double mean_us = static_cast<double>(total_us) / repetitions;
    const long max_us = *std::max_element(durations_us.begin(), durations_us.end());

    std::cout << label << " (" << repetitions << " repetitions):\n";
    std::cout << "  mean time:    " << mean_us << " us (" << mean_us / 1000.0 << " ms)\n";
    std::cout << "  max time:     " << max_us << " us (" << static_cast<double>(max_us) / 1000.0 << " ms)\n";
    std::cout << "  checksum: " << checksum << "\n";
}

// Runs the demo: generate, smooth, predict and print the results.
void test_loess();

// Application entry point: runs the demo and the benchmarks.
void run_app();
