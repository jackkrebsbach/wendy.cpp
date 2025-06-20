#ifndef WENDY_H
#define WENDY_H

#include <xtensor/containers/xarray.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

struct TestFunctionParams {
    const std::optional<int> number_test_functions;
    xt::xarray<int> radius_params = xt::pow(2, xt::xarray<double>{0, 1, 3.});
    double radius_min_time = 0.01;
    double radius_max_time = 5;

};

/**
 * @brief Weak form estimation of nonlinear dynamics
 */
class Wendy {
public:
    // User parameters for solving wendy system
    TestFunctionParams testFunctionParams;

    //Internal
    size_t D; //Dimension of system
    size_t J; //Number of parameters

    //
    xt::xarray<double> tt; // Time array (should be equispaced)
    xt::xarray<double> U; //Noisy data
    xt::xarray<double> V; //Orthonormal Test Function Matrix
    xt::xarray<double> V_prime; //Orthonormal derivative of Test Function Matrix

    std::vector<SymEngine::Expression> sym_system; //Symbolic representation of system
    std::vector<std::vector<SymEngine::Expression> > sym_system_jac;


    Wendy(const std::vector<std::string> &f,
          const xt::xarray<double> &U,
          const std::vector<float> &p0,
          const xt::xarray<double> &tt);

    static std::tuple<xt::xarray<double>, xt::xarray<double>> get_test_function_matrices();
    void log_details() const;
};

#endif // WENDY_H
