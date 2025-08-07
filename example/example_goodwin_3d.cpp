#include "../include/wendy/wendy.h"
#include <vector>
#include <string>
#include <xtensor/containers/xadapt.hpp>
#include <random>
#include <xtensor-blas/xlinalg.hpp>
#include <boost/numeric/odeint.hpp>

#include "../src/utils.h"

using namespace boost::numeric::odeint;

using state_type = std::vector<double>;

struct LorenzODE {
    std::vector<double> p;

    explicit LorenzODE(const std::vector<double> &p_) : p(p_) {}

    void operator()(const std::vector<double> &u, std::vector<double> &du_dt, double /*t*/) const {
        du_dt[0] = p[0] / (2.15 + p[2] * std::pow(u[2], p[3])) - p[1] * u[0];
        du_dt[1] = p[4] * u[0] - p[5] * u[1];
        du_dt[2] = p[6] * u[1] - p[7] * u[2];
    }
};


std::vector<std::vector<double> > add_noise( const std::vector<std::vector<double> > &data, const double noise_sd) {
    std::vector<std::vector<double> > noisy = data;
    const int n_points = data.size();
    const int dim = data[0].size();

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int d = 0; d < dim; ++d) {
        std::normal_distribution<> dist(0.0,  noise_sd);
        for (int i = 0; i < n_points; ++i) {
            noisy[i][d] += dist(gen);
        }
    }
    return noisy;
}

int main() {
    std::vector<double> p_star = {3.4884, 0.0969, 1, 10, 0.0969, 0.0581, 0.0969, 0.0775};
    std::vector<double> p0 = {3.0, 0.1, 4 , 12, 0.1, 0.1, 0.1, 0.1};
    const std::vector<double> u0 = {0.3617, 0.9137, 1.3934};
    std::vector u = u0;

    constexpr double noise_sd = 0.05;
    constexpr int num_samples = 100;
    constexpr double t0 = 0.0;
    constexpr double t1 = 80;

    const xt::xtensor<double, 1> t_eval = xt::linspace(t0, t1, num_samples);
    std::vector<state_type> u_eval;
    std::vector<double> t_vec;

    auto observer = [&](const state_type &x, double t) {
        u_eval.push_back(x);
        t_vec.push_back(t);
    };

    runge_kutta4<state_type> stepper;
    integrate_times(stepper, LorenzODE(p_star), u, t_eval.begin(), t_eval.end(), 0.01, observer);

    const auto u_noisy = add_noise(u_eval, noise_sd);

    const std::vector shape = {static_cast<size_t>(num_samples), u0.size()};

    std::vector<double> u_flat;
    for (const auto &row: u_noisy) {
        u_flat.insert(u_flat.end(), row.begin(), row.end());
    }

    const xt::xtensor<double,2> U = xt::adapt(u_flat, shape);

    const std::vector<std::string> system_eqs = {
        "p1 / (2.15 + p3 * u3^p4) - p2 * u1",
        "p5 * u1 - p6 * u2",
        "p7 * u2 - p8 * u3"
    };

    const xt::xtensor<double,1> tt = xt::linspace(t0, t1, num_samples);
    try {
       Wendy wendy(system_eqs, U, p0, tt, noise_sd);
       wendy.build_full_test_function_matrices();
       wendy.build_cost_function();
       //wendy.inspect_equations();
       wendy.optimize_parameters();

     // const auto mle = *wendy.obj;
     // std::cout << "\npstar: " << mle(std::vector<double>(p_star)) << std::endl;
     // std::cout << "p0:  " << mle(std::vector<double>(p0))  << std::endl; // pstar
     // std::cout << "   " <<  mle(std::vector<double>({2, 0.05, 1.5, 13, 0.15, 0.12, 0.18, 0.10}))  << std::endl;
     // std::cout << "   " <<mle(std::vector<double>({0.5, 0.15, 1.75, 7, 0.03, 0.03, 0.1, 0.08}))  << std::endl;
     // std::cout << "   " <<mle(std::vector<double>({0.25, 0.015, 3, 10, 0.1, 0.02, 0.15, 0.11}))  << std::endl;

    } catch (const std::exception &e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
    }
    return 0;
}