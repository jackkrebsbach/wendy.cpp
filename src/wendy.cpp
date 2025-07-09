#include "wendy.h"
#include "logger.h"
#include "utils.h"
#include "test_function.h"
#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>


Wendy::Wendy(const std::vector<std::string> &f, const xt::xtensor<double,2> &U, const std::vector<float> &p0, const xt::xtensor<double,1> &tt) {

    this->J = p0.size(); // Number of parameters in the system
    this->D = U.shape()[1]; // Dimension of the system
    this->U = U; // Noisy data
    this->tt = tt; // Time array
    this->sym_system = create_symbolic_system(f); // Symbolic representation of the RHS
    this->sym_system_jac = compute_jacobian(sym_system, create_symbolic_vars("p", J)); // Symbolic representation of the Jacobian of the RHS
    this->F = build_symbolic_system(sym_system, D, J); // callable function for numerics input

}


void Wendy::build_full_test_function_matrices(){

   auto radii = test_function_params.radius_params;
   // double min_radius = find_min_radius_int_error(U, tt,  2, 3, 100, 2);
    //TODO: Sanitize radius parameters to pass into the build_full_test_function_matrix

   enum TestFunctionOrder { VALUE = 0, DERIVATIVE = 1 };
   auto V = build_full_test_function_matrix(tt, radii, TestFunctionOrder::VALUE);
   auto V_prime = build_full_test_function_matrix(tt, radii, TestFunctionOrder::DERIVATIVE);

    if (!compute_svd) {
       this->V =V;
       this->V_prime = V_prime;
       return;
    }


    auto k_full = static_cast<double>(V.shape()[0]);// Number of rows, i.e. number of test functions
    auto mp1 = static_cast<double>(U.shape()[0]); //Number of observations
    double max_test_fun_matrix = test_function_params.k_max;
    int k_max = static_cast<int>(std::ranges::min({k_full, mp1, max_test_fun_matrix}));

    const auto SVD = xt::linalg::svd(V);
    xt::xtensor<double,1> singular_values = std::get<1>(SVD);
    xt::xtensor<double,2> U = std::get<0>(SVD);
    xt::xtensor<double,2> V_ = std::get<2>(SVD);


    xt::xtensor<double,1> condition_numbers = singular_values(0)/singular_values; // Recall the condition number of a matrix is σ_max/σ_min, these are ordered

    // TODO: if we compute the thin SVD we need look at the upper bound of the sum of singular values. So need to add more in. Look at Nic's code
    // TODO: How do we actually find the change point of the singular values?
    // We want to look at the change point of the cumulative sum of the singular values
    double sum_singular_values = xt::sum(singular_values)();

    // Natural information is the ratio of the first k singular values to the sum
    xt::xtensor<double,1> info_numbers = xt::zeros<double>({k_max});
    for (int i = 1; i < k_max ; i++ ) {
        info_numbers[i]= xt::sum(xt::view(singular_values, xt::range(0,i)))()/sum_singular_values;
    }

    auto k1 = find_last(condition_numbers,[this](const double x) { return x < test_function_params.max_test_fun_condition_number; } );
    auto k2 = find_last(info_numbers,[this](const double x) { return x < test_function_params.min_test_fun_info_number; } );

    if (k1 == -1) {k1 = std::numeric_limits<int>::max();}
    if (k2 == -1) {k2 =std::numeric_limits<int>::max();}

    auto K = std::min({k1,k2,k_max});

    if (K == k_max) {
        logger->warn("k_max is equal to k_max");
    }

    logger->info("Condition Number is now: {}", condition_numbers[K]);

    xt::xtensor<double,2> V_orthonormal = xt::view(V_, xt::all(), xt::range(0,K));

    this->V = xt::transpose(V_orthonormal);

    // TODO: Compare this to the Fourier Transformation and talk to Nic about this representation
    // Project the rows (each row is one test function) onto the O.N. basis of the test functions
    this->V_prime = xt::linalg::dot(V_prime, V_orthonormal);

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

