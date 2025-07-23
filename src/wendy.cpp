#include "symbolic_utils.h"
#include "utils.h"
#include "wendy.h"
#include "logger.h"
#include "test_function.h"
#include "objective/mle.h"

#include <symengine/expression.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>


Wendy::Wendy(const std::vector<std::string> &f_, const xt::xtensor<double, 2> &U_, const std::vector<double> &p0_,
             const xt::xtensor<double, 1> &tt_) :
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

    Jp_Jp_JU_f(build_T_f(
            build_symbolic_jacobian(
                build_symbolic_jacobian(Ju_f_symbolic, create_symbolic_vars("p", J)),
                create_symbolic_vars("p", J)
            ), D, J
        )
    ),

    // Variance of the data
    Sigma(xt::diag(xt::ones<double>({D}))) {
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

    const int min_radius_int_error = find_min_radius_int_error(U, tt, min_radius, max_radius);

    radii = radii * min_radius_int_error;
    radii = xt::filter(radii, radii < max_radius);

    enum TestFunctionOrder { VALUE = 0, DERIVATIVE = 1 };
    auto V = build_full_test_function_matrix(tt, radii, TestFunctionOrder::VALUE);
    auto V_prime = build_full_test_function_matrix(tt, radii, TestFunctionOrder::DERIVATIVE);

    if (!compute_svd) {
        this->V = V;
        this->V_prime = V_prime;
        return;
    }

    const auto k_full = static_cast<double>(V.shape()[0]);

    constexpr bool COMPUTE_FULL_MATRICES = false;
    const auto SVD = xt::linalg::svd(V, COMPUTE_FULL_MATRICES);
    const xt::xtensor<double, 2> U = std::get<0>(SVD);
    const xt::xtensor<double, 1> singular_values = std::get<1>(SVD);
    const xt::xtensor<double, 2> Vᵀ = std::get<2>(SVD);

    const double corner_index = static_cast<double>(get_corner_index(xt::cumsum(singular_values)));
    // Change point in cumulative sum of singular values
    const double max_test_fun_matrix = test_function_params.k_max; // Max # of test functions from user

    xt::xtensor<double, 1> condition_numbers = singular_values(0) / singular_values;
    // Recall the condition number of a matrix is σ_max/σ_min, σ_i are ordered

    double sum_singular_values = xt::sum(singular_values)();

    int k_max = static_cast<int>(std::ranges::min({k_full, mp1, max_test_fun_matrix, corner_index}));

    // Natural information is the ratio of the first k singular values to the sum
    xt::xtensor<double, 1> info_numbers = xt::zeros<double>({k_max});
    for (int i = 1; i < k_max; i++) {
        info_numbers[i] = xt::sum(xt::view(singular_values, xt::range(0, i)))() / sum_singular_values;
    }

    // Regularize with user input (hard max on condition # of test function & how much "information" we want)
    auto k1 = find_last(condition_numbers, [this](const double x) {
        return x < test_function_params.max_test_fun_condition_number;
    });
    auto k2 = find_last(info_numbers, [this](const double x) {
        return x > test_function_params.min_test_fun_info_number;
    });

    if (k1 == -1) { k1 = std::numeric_limits<int>::max(); }
    if (k2 == -1) { k2 = std::numeric_limits<int>::max(); }

    auto K = std::min({k1, k2, k_max});


    logger->info("Condition Number is now: {}", condition_numbers[K]);

    this->V = xt::view(Vᵀ, xt::range(0, K), xt::all());

    // TODO: Compare this to the Fourier Transformation
    // We have ϕ_full = UΣVᵀ =>  Vᵀ = Σ⁻¹ Uᵀϕ_full = ϕ. V has columns that form an O.N. for the row space of ϕ. Apply same transformation to ϕ' = Σ⁻¹ Uᵀϕ'_full
    const auto S_psuedo_inverse = xt::diag(1.0 / singular_values);
    const auto UtV_prime = xt::linalg::dot(xt::transpose(U), V_prime);

    this->V_prime = xt::view(xt::linalg::dot(S_psuedo_inverse, UtV_prime), xt::range(0, K), xt::all());
}

bool is_symmetric(const std::vector<std::vector<double> > &H, double tol = 1e-10) {
    const size_t n = H.size();
    for (size_t i = 0; i < n; ++i) {
        if (H[i].size() != n) return false; // Not square
        for (size_t j = 0; j < n; ++j) {
            if (std::abs(H[i][j] - H[j][i]) > tol) {
                std::cout << "Non-symmetric at (" << i << "," << j << "): "
                        << H[i][j] << " vs " << H[j][i] << std::endl;
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const std::vector<std::vector<double> > &mat, const int precision = 1) {
    const size_t n = mat.size();
    for (size_t i = 0; i < n; ++i) {
        for (const double j: mat[i]) {
            std::cout << std::setw(precision + 6) << std::setprecision(precision) << std::fixed << j << " ";
        }
        std::cout << std::endl;
    }
}

void Wendy::build_objective_function() const {
    const auto g = g_functor(F, V);
    const auto JU_g = J_g_functor(U, tt, V, Ju_f);
    const auto Jp_g = J_g_functor(U, tt, V, Jp_f);
    const auto Jp_JU_g = H_g_functor(U, tt, V, Jp_Ju_f);
    const auto Jp_Jp_g = H_g_functor(U, tt, V, Jp_Jp_f);
    const auto Jp_Jp_JU_g = T_g_functor(U, tt, V, Jp_Jp_JU_f);

    const auto L = CovarianceFactor(U, tt, V, V_prime, Sigma, JU_g, Jp_JU_g, Jp_Jp_JU_g);

    const auto b = xt::eval(-xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(V_prime, U)));

    const auto S_inv_r = S_inv_r_functor(L, g, b);


    // weak negative log-likelihood as a loss function
    const auto mle = MLE(U, tt, V, V_prime, L, g, b, JU_g, Jp_g, Jp_JU_g, Jp_Jp_g, Jp_Jp_JU_g, S_inv_r);

    const auto f = [&](const std::vector<double> &p) { return mle(p); }; // f
    const auto J_f = [&](const std::vector<double> &p) { return mle.Jacobian(p); }; // ∇f
    const auto H_f = [&](const std::vector<double> &p) { return mle.Hessian(p); }; // Hf (Hessian of f)

    print_matrix(H_f(p0));
}

void Wendy::log_details() const {
    logger->info("Wendy class details:");
    logger->info("  D (Number of state variables): {}", D);
    logger->info("  J (Number of parameters): {}", J);

    logger->info("  sym_system (Symbolic system expressions):");
    logger->info("    Size: {}", f_symbolic.size());
    for (size_t i = 0; i < f_symbolic.size(); ++i) {
        logger->info("      [{}]: {}", i, str(f_symbolic[i]));
    }

    logger->info("  sym_system_jac (Symbolic Jacobian):");
    logger->info("    Size: {}", Ju_f_symbolic.size());
    for (size_t i = 0; i < Ju_f_symbolic.size(); ++i) {
        std::string row;
        for (size_t j = 0; j < Ju_f_symbolic[i].size(); ++j) {
            row += str(Ju_f_symbolic[i][j]);
            if (j < Ju_f_symbolic[i].size() - 1)
                row += ", ";
        }
        logger->info("      Row {} (size {}): {}", i, Ju_f_symbolic[i].size(), row);
    }
}

