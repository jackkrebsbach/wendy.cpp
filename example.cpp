#include "src/wendy.h"
#include "src/logger.h"
#include <vector>
#include <string>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <random>
#include <cmath>
#include <xtensor-blas/xlinalg.hpp>

std::vector<double> goodwin_3d(double t, const std::vector<double> &u, const std::vector<double> &p) {
    double du1 = p[0] / (2.15 + p[2] * std::pow(u[2], p[3])) - p[1] * u[0];
    double du2 = p[4] * u[0] - p[5] * u[1];
    double du3 = p[6] * u[1] - p[7] * u[2];
    return {du1, du2, du3};
}

std::vector<std::vector<double> > integrate_goodwin(
    const std::vector<double> &u0,
    const std::vector<double> &p,
    double t0, double t1, int npoints) {
    int dim = u0.size();
    std::vector<std::vector<double> > result(npoints, std::vector<double>(dim));
    double dt = (t1 - t0) / (npoints - 1);
    std::vector<double> u = u0;
    double t = t0;

    for (int i = 0; i < npoints; ++i) {
        result[i] = u;
        auto du = goodwin_3d(t, u, p);
        for (int d = 0; d < dim; ++d) {
            u[d] += du[d] * dt;
        }
        t += dt;
    }
    return result;
}


std::vector<std::vector<double> > add_noise(
    const std::vector<std::vector<double> > &data,
    double noise_ratio) {
    std::vector<std::vector<double> > noisy = data;
    int npoints = data.size();
    int dim = data[0].size();

    std::vector<double> stddev(dim, 0.0);
    for (int d = 0; d < dim; ++d) {
        double mean = 0.0;
        for (int i = 0; i < npoints; ++i) mean += data[i][d];
        mean /= npoints;
        for (int i = 0; i < npoints; ++i) stddev[d] += std::pow(data[i][d] - mean, 2);
        stddev[d] = std::sqrt(stddev[d] / npoints);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int d = 0; d < dim; ++d) {
        std::normal_distribution<> dist(0.0, noise_ratio * stddev[d]);
        for (int i = 0; i < npoints; ++i) {
            noisy[i][d] += dist(gen);
        }
    }
    return noisy;
}

int main() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%^%l%$] %v");

    std::vector<double> p_star = {3.4884, 0.0969, 1, 10, 0.0969, 0.0581, 0.0969, 0.0775};
    const std::vector<double> u0 = {0.3617, 0.9137, 1.3934};

    constexpr int num_samples = 100;
    constexpr double t0 = 0.0;
    constexpr double t1 = 80.0;
    constexpr double noise_ratio = 0.15;
    const auto u_star = integrate_goodwin(u0, p_star, t0, t1, num_samples);
    const auto u_noisy = add_noise(u_star, noise_ratio);

    const std::vector shape = {static_cast<size_t>(num_samples), u0.size()};

    std::vector<double> u_flat;
    for (const auto &row: u_noisy) {
        u_flat.insert(u_flat.end(), row.begin(), row.end());
    }

    const xt::xtensor<double,2> U = xt::adapt(u_flat, shape);

    const std::vector<std::string> system_eqs = {
        "p0 / (2.15 + p2 * u2^p3) - p1 * u0",
        "p4 * u0 - p5 * u1",
        "p6 * u1 - p7 * u2"
    };

    const xt::xtensor<double,1> tt = xt::linspace(t0, t1, num_samples);
    const std::vector<double> p0(p_star.begin(), p_star.end());
    try {
       logger->set_level(spdlog::level::debug);

       Wendy wendy(system_eqs, U, p0, tt);
       wendy.build_full_test_function_matrices(); // Builds both full V and V_prime
       wendy.build_objective_function();

    } catch (const std::exception &e) {
        logger->error("Exception occurred: {}", e.what());
    }
    return 0;
}