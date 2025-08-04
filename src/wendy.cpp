#include "symbolic_utils.h"
#include "utils.h"
#include "wendy.h"
#include "test_function.h"
#include "cost/wnll.h"
#include "optimization/cpp_numerical.h"
#include "optimization/ceres.h"

#include <cppoptlib/solver/solver.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

Wendy::Wendy(const std::vector<std::string> &f_, const xt::xtensor<double, 2> &U_, const std::vector<double> &p0_,
    const xt::xtensor<double, 1> &tt_, double noise_sd, const bool compute_svd_):
    // Data
    tt(tt_),
    U(U_),
    p0(p0_),
    D(U_.shape()[1]),
    J(p0_.size()),

    // Symbolics
    f_symbolic(build_symbolic_f(f_)),
    Ju_f_symbolic(build_symbolic_jacobian(f_symbolic, create_symbolic_vars("u", D))),
    Jp_f_symbolic(build_symbolic_jacobian(f_symbolic, create_symbolic_vars("p", J))),

    // Callable functions
    f(build_f(f_symbolic, D, J)),
    F(f, U, tt),
    Ju_f(build_J_f(Ju_f_symbolic, D, J)),
    Jp_f(build_J_f(Jp_f_symbolic, D, J)),

    Ju_Ju_f(build_H_f(build_symbolic_jacobian(Ju_f_symbolic, create_symbolic_vars("u", D)), D, J)),
    Jp_Jp_f(build_H_f(build_symbolic_jacobian(Jp_f_symbolic, create_symbolic_vars("p", J)), D, J)),
    Jp_Ju_f(build_H_f(build_symbolic_jacobian(Ju_f_symbolic, create_symbolic_vars("p", J)), D, J)),
    Ju_Jp_f(build_H_f(build_symbolic_jacobian(Jp_f_symbolic, create_symbolic_vars("u", D)), D, J)),

    Jp_Jp_Ju_f(build_T_f(
            build_symbolic_jacobian(
                build_symbolic_jacobian(Ju_f_symbolic, create_symbolic_vars("p", J)),
                create_symbolic_vars("p", J)
            ), D, J
        )
    ),
    sigma(xt::eval(noise_sd*xt::ones<double>({D}))), // Standard deviation of the noise from the data
    compute_svd(compute_svd_) {
}

void Wendy::build_cost_function() {
    std::cout << "\n<< Initializing cost functions >>" << std::endl;
    g = std::make_unique<g_functor>(F, V);
    b = xt::eval(xt::ravel(xt::linalg::dot(-1.0*V_prime, U)));
    L = std::make_unique<Covariance>(U, tt, V, V_prime, sigma, Ju_f,Jp_f, Jp_Ju_f, Jp_Jp_Ju_f);
    cost = std::make_unique<WNLL>(U, tt, V, V_prime, *L, *g, b, Ju_f, Jp_f, Jp_Jp_f);
}

void Wendy::inspect_equations() const {

    if (!cost) { std::cout << "ERROR: Objective Function not Initialized" << std::endl; return; }

    std::cout << std::fixed << std::setprecision(3);

    const auto analytical_jacobian = cost->Jacobian(p0);
    const auto finite_jacobian = gradient_4th_order(*cost, p0);

    std::cout << "\nAnalytical Jacobian" << std::endl;
    for (const auto& row : analytical_jacobian) {
            std::cout << row << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nFinite Jacobian" << std::endl;
    for (const auto& row : finite_jacobian) {
            std::cout << row << " ";
        std::cout << std::endl; // Newline after each row
    }
    std::cout << std::endl;

    const auto analytical_hessian = cost->Hessian(p0);
    const auto finite_hessian = hessian_3rd_order(*cost, p0);

    std::cout << "\nAnalytical Hessian" << std::endl;
    for (const auto& row : analytical_hessian) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl; // Newline after each row
    }

    std::cout << "\n Finite Hessian" << std::endl;
    for (const auto& row : finite_hessian) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl; // Newline after each row
    }
}

void Wendy::optimize_parameters() {
    std::cout << "\n<< Optimizing parameters >>" << std::endl;

    if (!cost) { std::cout << "Warning: Objective Function not Initialized" << std::endl; return; }

    const auto mle = *cost;

    const MleProblem problem(mle);
    const Eigen::VectorXd x_init = Eigen::Map<const Eigen::VectorXd>(p0.data(), static_cast<int>(p0.size()));

    cppoptlib::solver::Lbfgs<MleProblem> solver;

    const auto initial_state = cppoptlib::function::FunctionState(x_init);

    auto [solution, solver_state] = solver.Minimize(problem, initial_state);

    std::cout << "\nSolver finished" << std::endl;
    std::cout << "Final Status: " << solver_state.status << std::endl;
    std::cout << "Found minimum at: " << solution.x.transpose() << std::endl;

    p_hat = std::vector<double>(solution.x.data(), solution.x.data() + solution.x.size());
}


void Wendy::build_full_test_function_matrices() {
    std::cout << "<< Building test matrices >>" << std::endl;
    const double dt = xt::mean(xt::diff(tt))();
    const int mp1 = static_cast<int>(U.shape()[0]); // Number of observations

    auto radii = test_function_params.radius_params;
    auto radius_min_time = test_function_params.radius_min_time;
    auto radius_max_time = test_function_params.radius_max_time;

    int min_radius = static_cast<int>(std::max(std::ceil(radius_min_time / dt), 2.0));
    int max_radius = static_cast<int>(std::floor(radius_max_time / dt)); // The diameter shouldn't be larger than the interior domain available

    int radius_min_max = static_cast<int>(std::floor(max_radius / xt::amax(radii)()));

    if (radius_min_max < min_radius) {
        radius_min_max = min_radius * 10;
    }

    if (int max_radius_for_interior = static_cast<int>(std::floor((mp1 - 2) / 2));
        max_radius > max_radius_for_interior) {
        max_radius = max_radius_for_interior;
    }

    std::cout << "  Min radius: " << min_radius << std::endl;
    std::cout << "  Max radius: " << max_radius << std::endl;
    std::cout << "  Minmax radius: " << radius_min_max << std::endl;

    const auto [ix, errors,radii_sweep ] = find_min_radius_int_error(U, tt, min_radius, radius_min_max);

    auto min_radius_int_error = radii_sweep[ix];

    this->min_radius_errors = errors;
    this->min_radius_radii = radii_sweep;
    this->min_radius_ix = ix;
    this->min_radius = min_radius_int_error;

    std::cout << "  Integral Error min radius: " << min_radius_int_error << std::endl;

    radii = test_function_params.radius_params * min_radius_int_error;

    this->radii = radii;

    auto radii_ = xt::eval(xt::filter(radii, radii < max_radius));

    if (radii_.size() == 0) {
        radii_ = xt::xtensor<int,1>({max_radius});
    }

    std::cout << "  Radii " << radii_ << std::endl;

    auto V_ = build_full_test_function_matrix(tt, radii_, 0);
    auto V_prime_ = build_full_test_function_matrix(tt, radii_, 1);

    if (!compute_svd) { this->V = V_; this->V_prime = V_prime_; return; }

    const auto k_full = static_cast<int>(V_.shape()[0]);

    std::cout << "  K Full: " << k_full << std::endl;

    const auto SVD = xt::linalg::svd(V_, false);

    const xt::xtensor<double, 2> U_ = std::get<0>(SVD);
    const xt::xtensor<double, 1> singular_values = std::get<1>(SVD);
    const xt::xtensor<double, 2> Vt = std::get<2>(SVD);

    double sum_singular_values = xt::sum(singular_values)();
    int k_max = std::min({ k_full, mp1, test_function_params.k_max });

    // Natural information is the ratio of the first k singular values to the sum
    xt::xtensor<double, 1> info_numbers = xt::zeros<double>({k_max});
    for (int i = 1; i < k_max; i++) { info_numbers[i] = xt::sum(xt::view(singular_values, xt::range(0, i)))() / sum_singular_values; }

    xt::xtensor<double, 1> condition_numbers = singular_values(0) / singular_values; // Recall the condition number of a matrix is σ_max/σ_min, σ_i are ordered

    // Regularize with user input (hard max on condition # of test function & how much "information" we want)
    auto k1 = find_last(condition_numbers, [this](const double x) {
        return x < test_function_params.max_test_fun_condition_number;
    });
    auto k2 = find_last(info_numbers, [this](const double x) {
        return x < test_function_params.min_test_fun_info_number;
    });

    if (k1 == -1) { k1 = std::numeric_limits<int>::max(); }
    if (k2 == -1) { k2 = std::numeric_limits<int>::max(); }

    auto K = std::min({k1, k2, k_max});

    std::cout << "  Condition Number is now: " << condition_numbers[K] <<std::endl;
    std::cout << "  Info Number is now: " << info_numbers[K] <<std::endl;
    std::cout << "  K is: " << K <<std::endl;


    this->V = xt::eval(xt::view(Vt, xt::range(0, K), xt::all()));
    // ϕ_full = UΣVᵀ =>  Vᵀ = Σ⁻¹ Uᵀϕ_full = ϕ.
    // Apply same transformation to ϕ' = Σ⁻¹ Uᵀϕ'_full
    std::cout << "  Calculating Vprime" << std::endl;

    const auto U_T = xt::eval(xt::transpose(U_));                   // (D, K)
    const auto UV = xt::eval(xt::linalg::dot(U_T, V_prime_));       // (D, mp1)
    const auto inv_s = xt::eval(1.0 / singular_values);             // (D,)
    const auto inv_s_view = xt::reshape_view(inv_s, std::vector<size_t>{inv_s.size(), 1});

    const auto scaled = xt::eval(UV * inv_s_view);

    this->V_prime = xt::eval(xt::view(scaled, xt::range(0, K), xt::all())); // (K, mp1)
}
