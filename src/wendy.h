#ifndef WENDY_H
#define WENDY_H


#include "objective/mle.h"
#include "weak_residual.h"
#include "weak_residual_covariance.h"
#include <symengine/expression.h>

struct TestFunctionParams {
    const std::optional<int> number_test_functions;
    // Number of test functions to use in the minimum radius selection process
    xt::xtensor<int, 1> radius_params = xt::pow(2, xt::xtensor<double, 1>{0, 1, 2, 3});
    // Radii to use for the test functions
    double radius_min_time = 0.01; // Minimum radius (in seconds)
    double radius_max_time = 5; // Maximum radius (in seconds)
    int k_max = 200; // Hard maximum on the number of test functions
    double max_test_fun_condition_number = 50; // Truncate the SVD of the test function matrices where this is true
    double min_test_fun_info_number = 0.95; // Double check where these come from
};

/**
 * @brief Weak form estimation of nonlinear dynamics (WENDy)
 */

class Wendy {
public:
    // Input Data
    xt::xtensor<double, 1> tt; // Time array (should be equispaced)
    xt::xtensor<double, 2> U; //Noisy data
    std::vector<double> p0; // Initial guess of the parameters

    // Internal
    size_t D; // Dimension of system
    size_t J; // Number of parameters
    xt::xtensor<double, 2> V; // Test Function Matrix
    xt::xtensor<double, 2> V_prime; //  Derivative of Test Function Matrix


    // Symbolics
    std::vector<SymEngine::Expression> f_symbolic; // Symbolic rhs u' = f(p,u,t)
    std::vector<std::vector<SymEngine::Expression> > Ju_f_symbolic; // Symbolic jacobian of the rhs w.r.t u⃗
    std::vector<std::vector<SymEngine::Expression> > Jp_f_symbolic; // Symbolic jacobian of the rhs w.r.t p⃗

    // Callable functions
    f_functor f; // u⃗' = f(p,u,t)
    F_functor F; // matrix valued function of rhs evaluation at all points F(p⃗, U, t⃗)

    J_f_functor Ju_f; // ∇ᵤf(p,u,t) Jacobian w.r.t state variable u⃗
    J_f_functor Jp_f; // ∇ₚf(p,u,t) Jacobian w.r.t parameters p⃗

    H_f_functor Ju_Ju_f; // Hᵤf(p,u,t) 3D Hessian w.r.t u⃗
    H_f_functor Jp_Jp_f; // Hₚf(p,u,t) 3D Hessian w.r.t p⃗
    H_f_functor Jp_Ju_f; // ∇ₚ∇ᵤf(p,u,t) 3D Tensor with mixed partials
    H_f_functor Ju_Jp_f; // ∇ᵤ∇ₚf(p,u,t) 3D Tensor with mixed partials

    T_f_functor Jp_Jp_Ju_f; // ∇ₚ∇ₚ∇ᵤf(p,u,t) 4D Tensor with mixed partials

    // Weak residual functors and related objects
    xt::xtensor<double, 2> Sigma; // Variance estimates for each dimension diagonal Matrix (D x D)

    // Input parameters for solving wendy system
    TestFunctionParams test_function_params;
    bool compute_svd = true; // If true then the test function matrices are orthonormal

    std::vector<double> p_hat;

    double min_radius{};
    xt::xtensor<double, 1> min_radius_errors;
    xt::xtensor<double, 1> min_radius_radii;
    xt::xtensor<double, 1> radii;
    size_t min_radius_ix{};


    // Weak Residual Equations
    std::unique_ptr<g_functor> g;
    std::unique_ptr<J_g_functor> Ju_g;
    std::unique_ptr<J_g_functor> Jp_g;
    std::unique_ptr<H_g_functor> Jp_Ju_g;
    std::unique_ptr<H_g_functor> Jp_Jp_g;
    std::unique_ptr<T_g_functor> Jp_Jp_Ju_g;
    std::unique_ptr<CovarianceFactor> L;
    xt::xtensor<double, 1> b;
    std::unique_ptr<S_inv_r_functor> S_inv_r;
    std::shared_ptr<MLE> obj;


    Wendy(const std::vector<std::string> &f_, const xt::xtensor<double, 2> &U_, const std::vector<double> &p0_,
          const xt::xtensor<double, 1> &tt_, double noise_sd = 0.05, bool compute_svd_ = true);

    void build_full_test_function_matrices();

    void build_objective_function();

    void inspect_equations() const;

    void optimize_parameters();
};

#endif // WENDY_H
