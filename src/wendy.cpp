#include "symbolic_utils.h"
#include "utils.h"
#include "wendy.h"
#include "test_function.h"
#include "objective/mle.h"
#include "optimization/ceres.h"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>



Wendy::Wendy(const std::vector<std::string> &f_, const xt::xtensor<double, 2> &U_, const std::vector<double> &p0_,
             const xt::xtensor<double, 1> &tt_, bool compute_svd_) :
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
    Ju_Jp_f(build_H_f(build_symbolic_jacobian(Jp_f_symbolic, create_symbolic_vars("u", D)), D, J)),
    Jp_Ju_f(build_H_f(build_symbolic_jacobian(Ju_f_symbolic, create_symbolic_vars("p", J)), D, J)),
    Jp_Jp_f(build_H_f(build_symbolic_jacobian(Jp_f_symbolic, create_symbolic_vars("p", J)), D, J)),

    Jp_Jp_JU_f(build_T_f(
            build_symbolic_jacobian(
                build_symbolic_jacobian(Ju_f_symbolic, create_symbolic_vars("p", J)),
                create_symbolic_vars("p", J)
            ), D, J
        )
    ),

    // Variance of the data
    Sigma(xt::diag(0.058*xt::ones<double>({D}))),
    compute_svd(compute_svd_) {}

void Wendy::build_objective_function() {

    const auto g = g_functor(F, V);
    const auto JU_g = J_g_functor(U, tt, V, Ju_f);
    const auto Jp_g = J_g_functor(U, tt, V, Jp_f);
    const auto Jp_JU_g = H_g_functor(U, tt, V, Jp_Ju_f);
    const auto Jp_Jp_g = H_g_functor(U, tt, V, Jp_Jp_f);
    const auto Jp_Jp_JU_g = T_g_functor(U, tt, V, Jp_Jp_JU_f);

    const auto L = CovarianceFactor(U, tt, V, V_prime, Sigma, JU_g, Jp_JU_g, Jp_Jp_JU_g);

    const auto b = xt::eval(-1*xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(V_prime, U)));

    const auto S_inv_r = S_inv_r_functor(L, g, b);

    const auto mle = MLE(U, tt, V, V_prime, L, g, b, JU_g, Jp_g, Jp_JU_g, Jp_Jp_g, Jp_Jp_JU_g, S_inv_r);


    const auto analytical_hessian = mle.Jacobian(p0);
    const auto finite_hessian = gradient_4th_order(mle, p0);

    std::cout << "\nAnalytical Hessian" << std::endl;
    // for (const auto& row : analytical_hessian) {
        for (const auto& val : analytical_hessian) //{
            std::cout << val << " ";
        // }
        std::cout << std::endl; // Newline after each row
    // }

    std::cout << std::endl;

    std::cout << "\n Finite Hessian" << std::endl;
    // for (const auto& row : finite_hessian) {
        for (const auto& val : finite_hessian) //{
            std::cout << val << " ";
        // }
        // std::cout << std::endl; // Newline after each row
    // }

    std::cout << std::endl;

    MyMLEProblem problem(mle);
    Eigen::VectorXd x_init = Eigen::Map<const Eigen::VectorXd>(p0.data(), p0.size());

    cppoptlib::solver::Lbfgs<MyMLEProblem> solver;

    auto initial_state = cppoptlib::function::FunctionState(x_init);

    // solver.SetCallback(cppoptlib::solver::PrintProgressCallback<MyMLEProblem, decltype(initial_state)>(std::cout));

    auto [solution, solver_state] = solver.Minimize(problem, initial_state);

    std::cout << "\nSolver finished!" << std::endl;
    std::cout << "Final Status: " << solver_state.status << std::endl;
    std::cout << "Found minimum at: " << solution.x.transpose() << std::endl;

    p_hat = std::vector<double>(solution.x.data(), solution.x.data() + solution.x.size());
}


void Wendy::build_full_test_function_matrices() {
    const double dt = xt::mean(xt::diff(tt))();
    const auto mp1 = static_cast<double>(U.shape()[0]); // Number of observations


    auto radii = test_function_params.radius_params;
    auto radius_min_time = test_function_params.radius_min_time;
    auto radius_max_time = test_function_params.radius_max_time;

    int min_radius = static_cast<int>(std::max(std::ceil(radius_min_time / dt), 2.0)); // At least two data points

    // The diameter shouldn't be larger than the interior domain available
    int max_radius = static_cast<int>(std::floor(radius_max_time / dt));
    if (int max_radius_for_interior = static_cast<int>(std::floor((mp1 - 2) / 2));
        max_radius > max_radius_for_interior) {
        max_radius = max_radius_for_interior;
    }

    const auto [ix, errors, _] = find_min_radius_int_error(U, tt, min_radius, max_radius);

    const int min_radius_int_error = _[ix];

    this->min_radius_errors = errors;
    this->min_radius_radii = _;
    this->min_radius_ix = ix;
    this->min_radius = min_radius_int_error;

    radii = radii * min_radius_int_error;
    const auto radii_ = xt::filter(radii, radii < max_radius);

    auto V = build_full_test_function_matrix(tt, radii_, 0);
    auto V_prime = build_full_test_function_matrix(tt, radii_, 1);

    if (!compute_svd) {
        this->V = V;
        this->V_prime = V_prime;
        return;
    }

    const auto k_full = static_cast<double>(V.shape()[0]);

    constexpr bool COMPUTE_FULL_MATRICES = false;
    const auto SVD = xt::linalg::svd(V, COMPUTE_FULL_MATRICES);
    const xt::xtensor<double, 2> U_ = std::get<0>(SVD);
    const xt::xtensor<double, 1> singular_values = std::get<1>(SVD);
    const xt::xtensor<double, 2> Vᵀ = std::get<2>(SVD);

    const double corner_index = static_cast<double>(get_corner_index(xt::cumsum(singular_values))); // Change point in cumulative sum of singular values
    const double max_test_fun_matrix = test_function_params.k_max; // Max # of test functions from user
    double sum_singular_values = xt::sum(singular_values)();

    int k_max = static_cast<int>(std::ranges::min({k_full, mp1, max_test_fun_matrix }));

    // Natural information is the ratio of the first k singular values to the sum
    xt::xtensor<double, 1> info_numbers = xt::zeros<double>({k_max});
    for (int i = 1; i < k_max; i++) {
        info_numbers[i] = xt::sum(xt::view(singular_values, xt::range(0, i)))() / sum_singular_values;
    }

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

    std::cout << "Condition Number is now: " << condition_numbers[K] <<std::endl;
    std::cout << "Info Number is now: " << info_numbers[K] <<std::endl;

    this->V = xt::eval(xt::view(Vᵀ, xt::range(0, K), xt::all()));

    // We have ϕ_full = UΣVᵀ =>  Vᵀ = Σ⁻¹ Uᵀϕ_full = ϕ. V has columns that form an O.N. for the row space of ϕ. Apply same transformation to ϕ' = Σ⁻¹ Uᵀϕ'_full
    const auto UtV_prime = xt::linalg::dot(xt::transpose(U_), V_prime);

    this->V_prime = xt::eval(xt::view(xt::linalg::dot( xt::diag(1.0 / singular_values), UtV_prime), xt::range(0, K), xt::all()));
}
