#include "src/wendy.h"
#include <vector>
#include <string>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <random>
#include <cmath>
#include <xtensor-blas/xlinalg.hpp>

std::vector<double> logistic(double t, const std::vector<double> &u, const std::vector<double> &p) {
    double du1 = u[0]*p[0]- p[1]*std::pow(u[0],2);
    return {du1};
}

std::vector<std::vector<double> > integrate_(
    const std::vector<double> &u0,
    const std::vector<double> &p,
    double t0, double t1, int npoints) {
    int dim = u0.size();
    std::vector<std::vector<double> > result(npoints, std::vector<double>(dim));
    const double dt = (t1 - t0) / (npoints - 1);
    std::vector<double> u = u0;
    double t = t0;

    for (int i = 0; i < npoints; ++i) {
        result[i] = u;
        auto du = logistic(t, u, p);
        for (int d = 0; d < dim; ++d) {
            u[d] += du[d] * dt;
        }
        t += dt;
    }
    return result;
}

std::vector<std::vector<double>> add_noise(
    const std::vector<std::vector<double>>& data,
    double noise_ratio) {
    std::vector<std::vector<double>> noisy = data;
    int npoints = data.size();
    int dim = data[0].size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    for (int i = 0; i < npoints; ++i) {
        for (int d = 0; d < dim; ++d) {
            noisy[i][d] += noise_ratio * dist(gen);
        }
    }
    return noisy;
}


int main() {

    std::vector<double> p_star = {1.0, 1.0};
    std::vector<double> p_perturbed = {1.5 ,1.5};

    const std::vector<double> u0 = {0.01};

    constexpr double noise_ratio = 0.05;
    constexpr int num_samples = 100;
    constexpr double t0 = 0.0;
    constexpr double t1 = 1.0;

    const auto u_star = integrate_(u0, p_perturbed, t0, t1, num_samples);

    const auto u_noisy = add_noise(u_star, noise_ratio);

    const std::vector shape = {static_cast<size_t>(num_samples), u0.size()};

    std::vector<double> u_flat;
    for (const auto &row: u_noisy) {
        u_flat.insert(u_flat.end(), row.begin(), row.end());
    }

    const xt::xtensor<double,2> U = xt::adapt(u_flat, shape);

    const std::vector<std::string> system_eqs = {
        "u1*p1 - p2*u1^2"
    };

    const xt::xtensor<double,1> tt = xt::linspace(t0, t1, num_samples);
    const std::vector<double> p0(p_perturbed.begin(), p_perturbed.end());
    try {

       Wendy wendy(system_eqs, U, p0, tt, true);
       wendy.build_full_test_function_matrices(); // Builds both full V and V_prime
       wendy.build_objective_function();

    } catch (const std::exception &e) {
        std::cout << "Exception occurred: {}" << e.what() << std::endl;
    }
    return 0;
}
