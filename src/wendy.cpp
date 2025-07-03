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

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(xtensor_matrix_to_eigen(V),Eigen::ComputeThinU | Eigen::ComputeThinV); //Check how fast this is
    xt::xarray<double> singular_values = eigen_to_xtensor_1d( svd.singularValues());

    auto k_full = static_cast<double>(V.shape()[0]);// Number of rows, i.e. number of test functions
    auto mp1 = static_cast<double>(U.shape()[0]); //Number of observations
    double max_test_fun_matrix = test_function_params.k_max;
    int k_max = static_cast<int>(std::ranges::min({k_full, mp1, max_test_fun_matrix}));

    //Recall the condition number of a matrix is σ_max/σ_min, these are ordered
    xt::xarray<double> condition_numbers = singular_values(0)/singular_values;

    // We want to look at the change point of the cumulative sum of the singular values
    // TODO: if we compute the thin SVD we need look at the upper bound of the sum of singular values
    // So probably need to add more in. Look at Nic's code
    double sum_singular_values = xt::sum(singular_values)();

    // Natural information is the ratio of the first k singular values to the sum
    xt::xarray<double> info_numbers = xt::zeros<double>({k_max});
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
    auto V_orthonormal = xt::view(eigen_matrix_to_xtensor(svd.matrixV()), xt::all(), xt::range(0,K));

    // We want the columns of Vprime to be in the basis of the SVD
    auto Vp_orthonormal = project_onto_svd_basis(V_prime,eigen_matrix_to_xtensor(svd.matrixU()), singular_values);

    this->V = xt::transpose(V_orthonormal);
    this->V_prime = xt::view(Vp_orthonormal, xt::range(0,K), xt::all());
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

