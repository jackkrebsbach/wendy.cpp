#include "../include/wendy/wendy.h"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include <xtensor/containers/xadapt.hpp>
#include <random>
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

using state_type = std::vector<double>;


std::vector<std::vector<double> > add_noise(const std::vector<std::vector<double> > &data, const double noise_sd) {
    std::vector<std::vector<double> > noisy = data;
    const int n_points = static_cast<int>(data.size());
    const int dim = static_cast<int>(data[0].size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    for (int i = 0; i < n_points; ++i) {
        for (int d = 0; d < dim; ++d) {
             // noisy[i][d] += noise_sd * dist(gen);
             noisy[i][d] *= std::exp(noise_sd * dist(gen));
        }
    }
    return noisy;
}

struct LogisticODE {
    std::vector<double> p;

    explicit LogisticODE(const std::vector<double> &p_) : p(p_) {
    }

    void operator()(const state_type &u, state_type &du_dt, double /*t*/) const {
        du_dt[0] = u[0] * p[0] - p[1] * std::pow(u[0], 2);
    }
};

int main() {
    const std::vector p_star = {1.0, 1.0};
    std::vector p0 = {0.5, 0.5};

    const std::vector u0 = {0.005};
    std::vector u = u0;

    constexpr double noise_sd = 0.05;
    constexpr int num_samples = 101;

    constexpr double t0 = 0.0;
    constexpr double t1 = 10.0;

    const xt::xtensor<double, 1> t_eval = xt::linspace(t0, t1, num_samples);
    std::vector<state_type> u_eval;
    std::vector<double> t_vec;

    auto observer = [&](const state_type &x, double t) {
        u_eval.push_back(x);
        t_vec.push_back(t);
    };

    runge_kutta4<state_type> stepper;
    integrate_times(stepper, LogisticODE(p_star), u, t_eval.begin(), t_eval.end(), 0.001, observer);

    const auto u_noisy = add_noise(u_eval, noise_sd);

    const std::vector shape = {static_cast<size_t>(num_samples), u0.size()};

    std::vector<double> u_flat;
    for (const auto &row: u_noisy) {
        u_flat.insert(u_flat.end(), row.begin(), row.end());
    }

    const xt::xtensor<double, 2> U = xt::adapt(u_flat, shape);

    try {

        const std::vector<std::string> system_eqs = {"u1*p1 - p2*u1^2"};
        const xt::xtensor<double, 1> tt = xt::linspace(t0, t1, num_samples);

        Wendy wendy(system_eqs, U, p0, tt, noise_sd, true , "LogNormal");
        wendy.build_full_test_function_matrices();
        wendy.build_cost_function();
        // wendy.inspect_equations();
        wendy.optimize_parameters();

        // auto wnll = *wendy.cost;
        // auto S = wnll.S(p0);
        // std::cout << "Condition Number: " << xt::linalg::cond(S, 2) << std::endl;

        // std::cout << "\n p0" << std::endl;
        // std::cout << "\n pstar: " << mle(p_star) << std::endl;
        // std::cout << std::endl;
        // std::cout << mle(std::vector<double>(p0))  << std::endl;
        // std::cout << mle(std::vector<double>({0.55, 0.55}))  << std::endl;
        // std::cout << mle(std::vector<double>({1.5, 1.5}))  << std::endl;
        // std::cout << mle(std::vector<double>({1.25, 1.25}))  << std::endl;

    } catch (const std::exception &e) {
        std::cout << "Error occurred: " << e.what() << std::endl;
    }
    return 0;
}