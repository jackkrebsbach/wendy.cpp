#ifndef WENDY_H
#define WENDY_H

#include "utils.h"
#include <xtensor/views/xview.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

struct TestFunctionParams {
    const std::optional<int> number_test_functions; // Number of test functions to use in the minimum radius selection process
    xt::xtensor<int,1> radius_params = xt::pow(2, xt::xtensor<double,1>{0,1,3}); // Radii to use for the test functions
    double radius_min_time = 0.01; // Minimum radius (in seconds)
    double radius_max_time = 5; // Maximum radius (in seconds)
    int k_max = 200; // Hard maximum on the number of test functions
    double max_test_fun_condition_number = 1e4; // Truncate the SVD of the test function matrices where this is true
    double min_test_fun_info_number = 0.95; // Double check where these come from
};

/**
 * @brief Weak form estimation of nonlinear dynamics (WENDy)
 */
class Wendy {
public:
    // Input Data
    xt::xtensor<double,1> tt; // Time array (should be equispaced)
    xt::xtensor<double, 2> U; //Noisy data
    // Internal
    size_t D; // Dimension of system
    size_t J; // Number of parameters
    xt::xtensor<double,2> V; // Test Function Matrix
    xt::xtensor<double,2> V_prime; //  Derivative of Test Function Matrix
    std::vector<SymEngine::Expression> f_symbolic; // Symbolic rhs u' = f(p,u,t)
    std::vector<std::vector<SymEngine::Expression>> Ju_f_symbolic; // Symbolic jacobian of the rhs
    f f; // u' = f(p,u,t)
    Ju_f Ju_f; // J_uf(p,u,t) Jacobian w.r.t state variable u

    // Input parameters for solving wendy system
    TestFunctionParams test_function_params;
    bool compute_svd = true; // If true then the test function matrices are orthonormal

    Wendy(const std::vector<std::string> &f,
        const xt::xtensor<double,2> &U,
        const std::vector<float> &p0,
        const xt::xtensor<double,1> &tt);

    void  build_full_test_function_matrices();
    void log_details() const;

    [[nodiscard]] const xt::xtensor<double,2>& getV() const { return this->V; }
    [[nodiscard]] const xt::xtensor<double,2>& getV_prime() const { return this->V_prime; }
};

#endif // WENDY_H