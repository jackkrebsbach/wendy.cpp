#include "wendy.h"
#include "logger.h"
#include "test_function.h"
#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>


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


xt::xarray<double> Wendy::build_full_test_function_matrix(const int order){

   auto radii = testFunctionParams.radius_params;

   double min_radius = find_min_radius_int_error(U, tt,  2, 3, 100, 2);

    // Vector containing test matrices for one radius
   std::vector<xt::xarray<double>> test_matrices;
   for (int i = 0; i < radii.shape()[0]; ++i) {
        // Build the test matrix for one radius
        xt::xarray<double> V_k = build_test_function_matrix(tt, radii[i], order);
        test_matrices.emplace_back(std::move(V_k));
    }
    xt::xarray<double> V_full = test_matrices[0];

    for (size_t i = 1; i < test_matrices.size(); ++i) {
        //Subtle bug: Must wrap concatenation in xt::xarray<double>(...) otherwise we get data loss
        V_full = xt::xarray<double>(xt::concatenate(xt::xtuple(V_full, test_matrices[i]),0));
    }

    // Not really a good idea but will do this for now
    if (order == 0) {
        this->V=V_full;
    } else {
        this->V_prime=V_full;
    }
    return(V_full);
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

