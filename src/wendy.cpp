#include "wendy.h"
#include "logger.h"
#include "utils.h"
#include "test_function.h"
#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>


Wendy::Wendy(const std::vector<std::string> &f, const xt::xtensor<double,2> &U, const std::vector<float> &p0, const xt::xtensor<double,1> &tt) {
    this->J = p0.size(); // Number of parameters in the system
    this->D = U.shape()[1]; // Dimension of the system
    this->U = U; // Noisy data (for now assumed to be additive Gaussian)
    this->tt = tt; // Time array (should be equispaced)
    this->sym_system = create_symbolic_system(f); // Symbolic representation of the RHS
    this->sym_system_jac = compute_jacobian(sym_system, create_symbolic_vars("p", J)); // Symbolic representation of the Jacobian of the RHS
    this->F = build_symbolic_system(sym_system, D, J); // callable function for numerics input
}


void Wendy::build_full_test_function_matrices(){

   const double dt =xt::mean(xt::diff(tt))();
   const auto k_full = static_cast<double>(V.shape()[0]);// Number of test functions (# of rows)
   const auto mp1 = static_cast<double>(U.shape()[0]); // Number of observations

   auto radii = test_function_params.radius_params; // Radii multipliers
   auto radius_min_time = test_function_params.radius_min_time;
   auto radius_max_time = test_function_params.radius_max_time;

   int min_radius = static_cast<int>(std::max(std::ceil(radius_min_time/dt), 2.0)); // At least two data points

   // The diameter shouldn't be larger than the interior domain available
    int max_radius = static_cast<int>(std::floor(radius_max_time/dt));
    if(int max_radius_for_interior = static_cast<int>(std::floor((mp1-2)/2)); max_radius > max_radius_for_interior) {
        max_radius =  max_radius_for_interior;
    }

   const int min_radius_int_error = find_min_radius_int_error(U, tt, min_radius, max_radius);

   radii = radii*min_radius_int_error;
   radii = xt::filter(radii, radii < max_radius);

   enum TestFunctionOrder { VALUE = 0, DERIVATIVE = 1 };
   auto V = build_full_test_function_matrix(tt, radii, TestFunctionOrder::VALUE);
   auto V_prime = build_full_test_function_matrix(tt, radii, TestFunctionOrder::DERIVATIVE);

   if (!compute_svd) {this->V =V; this->V_prime = V_prime; return;}

   constexpr bool COMPUTE_FULL_MATRICES = false;
   const auto SVD = xt::linalg::svd(V, COMPUTE_FULL_MATRICES);
   const xt::xtensor<double,2> U = std::get<0>(SVD);
   const xt::xtensor<double,1> singular_values = std::get<1>(SVD);
   const xt::xtensor<double,2> Vᵀ = std::get<2>(SVD);

   const double corner_index =  static_cast<double>(get_corner_index(xt::cumsum(singular_values))); // Change point in cumulative sum of singular values
   const double max_test_fun_matrix = test_function_params.k_max; // Max # of test functions from user

   xt::xtensor<double,1> condition_numbers = singular_values(0)/singular_values; // Recall the condition number of a matrix is σ_max/σ_min, σ_i are ordered

   double sum_singular_values = xt::sum(singular_values)();

   int k_max = static_cast<int>(std::ranges::min({k_full, mp1, max_test_fun_matrix, corner_index}));

   // Natural information is the ratio of the first k singular values to the sum
    xt::xtensor<double,1> info_numbers = xt::zeros<double>({k_max});
    for (int i = 1; i < k_max ; i++ ) {
        info_numbers[i]= xt::sum(xt::view(singular_values, xt::range(0,i)))()/sum_singular_values;
    }

    // Regularize with user input (hard max on condition # of test function & how much "information" we want)
    auto k1 = find_last(condition_numbers,[this](const double x) { return x < test_function_params.max_test_fun_condition_number; } );
    auto k2 = find_last(info_numbers,[this](const double x) { return x > test_function_params.min_test_fun_info_number; } );

    if (k1 == -1) {k1 = std::numeric_limits<int>::max();}
    if (k2 == -1) {k2 =std::numeric_limits<int>::max();}

    auto K = std::min({k1,k2,k_max});

    logger->info("Condition Number is now: {}", condition_numbers[K]);

    this-> V= xt::view(Vᵀ, xt::range(0,K), xt::all());

    // TODO: Compare this to the Fourier Transformation
    // We have ϕ_full = UΣVᵀ =>  Vᵀ = Σ⁻¹ Uᵀϕ_full = ϕ. V has columns that form an O.N. for the row space of ϕ
    // Apply same transformation to ϕ' = Σ⁻¹ Uᵀϕ'_full

    const auto Σ_psuedo_inverse = xt::diag(1.0/singular_values);
    const auto UᵀV_prime =xt::linalg::dot(xt::transpose(U), V_prime);

    this->V_prime = xt::view(xt::linalg::dot(Σ_psuedo_inverse ,UᵀV_prime), xt::range(0,K), xt::all());
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

