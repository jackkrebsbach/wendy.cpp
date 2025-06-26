#include "wendy.h"
#include "logger.h"
#include "utils.h"
#include "test_function.h"
#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>
#include <Eigen/Dense>
#include <xtensor/containers/xarray.hpp>



Wendy::Wendy(const std::vector<std::string> &f, const xt::xarray<double> &U, const std::vector<float> &p0, const xt::xarray<double> &tt) {
    if (U.dimension() != 2) {
        throw std::invalid_argument("U must be 2-dimensional");
    }

    J = p0.size(); // Number of parameters in the system
    D = U.shape()[1]; // Dimension of the system

    this->U = U; //Noisy data
    this->tt = tt; // Time array

    sym_system = create_symbolic_system(f);

    const auto p_symbols = create_symbolic_vars("p", J);
    const auto grad_p_f = compute_jacobian(sym_system, p_symbols);

    sym_system_jac = grad_p_f;
}


void Wendy::build_full_test_function_matrices(){

   auto radii = test_function_params.radius_params;
   double min_radius = find_min_radius_int_error(U, tt,  2, 3, 100, 2);

   auto V = build_full_test_function_matrix(tt, radii, 0);
   auto V_prime = build_full_test_function_matrix(tt, radii, 1);

    if (!compute_svd) {
       this->V =V;
       this->V_prime = V_prime;
       return;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(xtensor_matrix_to_eigen(V)); //Check how fast this is
    Eigen::VectorXd singular_values = svd.singularValues();

    auto k_full = V.shape()[0];// Number of rows ie number of test functions
    auto mp1 = U.shape()[0]; //Number of observations

    //Recall the condition number of a matrix is (σ_max)/(σ_min), these are ordered
    auto condition_numbers = singular_values[0]/singular_values.array();
    //We want to look at the change point of the cumulative sum of the singular values

    auto sum_singular_values = singular_values.sum();




}


void Wendy::log_details() const {
    logger->info("Wendy class details:");
    logger->info("  D (Number of state variables): {}", D);
    logger->info("  J (Number of parameters): {}", J);

    logger->info("  sym_system (Symbolic system expressions):");
    logger->info("    Size: {}", sym_system.size());
    for (size_t i = 0; i < sym_system.size(); ++i) {
        logger->info("      [{}]: {}", i, str(sym_system[i]));
    }


    logger->info("  sym_system_jac (Symbolic Jacobian):");
    logger->info("    Size: {}", sym_system_jac.size());
    for (size_t i = 0; i < sym_system_jac.size(); ++i) {
        std::string row;
        for (size_t j = 0; j < sym_system_jac[i].size(); ++j) {
            row += str(sym_system_jac[i][j]);
            if (j < sym_system_jac[i].size() - 1)
                row += ", ";
        }
        logger->info("      Row {} (size {}): {}", i, sym_system_jac[i].size(), row);
    }
}

