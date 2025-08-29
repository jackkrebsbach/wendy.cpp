#include "../src/wendy.h"
#include "../src/utils.h"

#include <vector>
#include <string>
#include <xtensor/containers/xadapt.hpp>
#include <random>
#include <xtensor-blas/xlinalg.hpp>
#include <boost/numeric/odeint.hpp>


using namespace boost::numeric::odeint;

using state_type = std::vector<double>;

struct Goodwin_2D {
    std::vector<double> p;

    explicit Goodwin_2D(const std::vector<double> &p_) : p(p_) {}

    void operator()(const std::vector<double> &u, std::vector<double> &du_dt, double /*t*/) const {
        du_dt[0] = p[0] * u[1] - p[1]* std::pow(u[0], 3) + p[2]*std::pow(u[0],2) - p[3]*u[2];
        du_dt[1] = p[4] - p[5] * std::pow(u[0],2) + p[6]*u[1];
        du_dt[2] = p[7]*u[0] + p[8] - p[9]*u[2];
    }
};

    const std::vector<std::string> system_eqs = {
        "p1*u2 - p2*u1^3 + p3*u1^2 - p4*u3",
        "p5 - p6*u1^2 + p7*u2",
        "p8*u1 + p9 - p10*u3"
    };


std::vector<std::vector<double> > add_noise( const std::vector<std::vector<double> > &data, const double noise_sd) {
    std::vector<std::vector<double> > noisy = data;
    const int n_points = data.size();
    const int dim = data[0].size();

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int d = 0; d < dim; ++d) {
        std::normal_distribution<> dist(0.0,  1.0);
        for (int i = 0; i < n_points; ++i) {
            noisy[i][d] += noise_sd * dist(gen);
            // noisy[i][d] *= std::exp(noise_sd * dist(gen));

        }
    }
    return noisy;
}

int main() {
    std::vector<double> p_star = {10, 10, 30, 10, 10, 50, 10, 0.04, 0.0319, 0.01};

    std::vector<double> p0 = {15, 16, 45, 5, 6, 55, 0.3, 0.5, 0.6, 0.1 };
    const std::vector<double> u0 = {-1.31, -7.6,-0.2};

    std::vector u = u0;

    constexpr double noise_sd = 0.05;
    constexpr int num_samples = 100;
    constexpr double t0 = 0.0;
    constexpr double t1 = 10;

    const xt::xtensor<double, 1> t_eval = xt::linspace(t0, t1, num_samples);
    std::vector<state_type> u_eval;
    std::vector<double> t_vec;

    auto observer = [&](const state_type &x, double t) {
        u_eval.push_back(x);
        t_vec.push_back(t);
    };

    runge_kutta4<state_type> stepper;
    integrate_times(stepper, Goodwin_2D(p_star), u, t_eval.begin(), t_eval.end(), 0.00001, observer);

    const auto u_noisy = add_noise(u_eval, noise_sd);

    const std::vector shape = {static_cast<size_t>(num_samples), u0.size()};

    std::vector<double> u_flat;
    for (const auto &row: u_noisy) {
        u_flat.insert(u_flat.end(), row.begin(), row.end());
    }

    const xt::xtensor<double,2> U = xt::adapt(u_flat, shape);
    const xt::xtensor<double,1> tt = xt::linspace(t0, t1, num_samples);

    try {

       Wendy wendy(system_eqs, U, p0, tt, "info");
       wendy.build_full_test_function_matrices();
       wendy.build_cost_function();
       // wendy.inspect_equations();
       // wendy.optimize_parameters();

        std::cout << U << std::endl;

     // const auto cost = *wendy.cost;
     // std::cout << "\np_star: " << cost() << std::endl;
     // std::cout << "   " << cost(std::vector<double>({44.9581, 1.70535, 1.41093, 0.979979, 1.13597}))  << std::endl;
     // std::cout << "p0:  " << cost(std::vector<double>(p0))  << std::endl; // pstar
     // std::cout << "   " << cost(std::vector<double>({ 66, 2.6, 3.0, 1.6, 2}))  << std::endl;

    } catch (const std::exception &e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
    }
    return 0;
}