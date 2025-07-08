#ifndef WENDY_H
#define WENDY_H

#include <xtensor/containers/xarray.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>


struct TestFunctionParams {
    const std::optional<int> number_test_functions; // Number of test functions to use in the minimum radius selection process
    xt::xarray<int> radius_params = xt::pow(2, xt::xarray<double>{1,3}); // Radii to use for the test functions
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
    // User parameters for solving wendy system
    TestFunctionParams test_function_params;

    // Input parameters
    bool compute_svd = true; // If true then the test function matrices are orthonormal

    // Input Data
    xt::xarray<double> tt; // Time array (should be equispaced)
    xt::xarray<double> U; //Noisy data

    // Internal
    std::vector<SymEngine::LambdaRealDoubleVisitor> F; // the RHS symbolic system
    size_t D; // Dimension of system
    size_t J; // Number of parameters
    xt::xarray<double> V; // Test Function Matrix (can be orthonormal)
    xt::xarray<double> V_prime; //  Derivative of Test Function Matrix (can be orthonormal)


    std::vector<SymEngine::Expression> sym_system; //Symbolic representation of system
    std::vector<std::vector<SymEngine::Expression> > sym_system_jac;


    Wendy(const std::vector<std::string> &f,
        const xt::xarray<double> &U,
        const std::vector<float> &p0,
        const xt::xarray<double> &tt);

    void  build_full_test_function_matrices();
    void log_details() const;

    //Getters
    [[nodiscard]] const xt::xarray<double>& getV() const { return V; }
    [[nodiscard]] const xt::xarray<double>& getV_prime() const { return V_prime; }
};

#endif // WENDY_H
