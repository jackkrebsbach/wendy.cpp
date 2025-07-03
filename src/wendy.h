#ifndef WENDY_H
#define WENDY_H

#include <xtensor/containers/xarray.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

#include "test_function.h"

struct TestFunctionParams {
    const std::optional<int> number_test_functions;
    xt::xarray<int> radius_params = xt::pow(2, xt::xarray<double>{1,3});
    double radius_min_time = 0.01;
    double radius_max_time = 5;
    int k_max = 200;
    double max_test_fun_condition_number = 1e-4;
    double min_test_fun_info_number = 0.95; //Double check where these come from
};

/**
 * @brief Weak form estimation of nonlinear dynamics (WENDy)
 */
class Wendy {
public:
    // User parameters for solving wendy system
    TestFunctionParams test_function_params;

    // Input parameters
    bool compute_svd = true;

    // Input Data
    xt::xarray<double> tt; // Time array (should be equispaced)
    xt::xarray<double> U; //Noisy data

    //Internal
    size_t D; //Dimension of system
    size_t J; //Number of parameters
    xt::xarray<double> V; //Orthonormal Test Function Matrix
    xt::xarray<double> V_prime; //Orthonormal derivative of Test Function Matrix


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
};

#endif // WENDY_H
