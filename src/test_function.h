#pragma once
#include <xtensor/containers/xarray.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

//Default test function used, eta controls the shape
double phi(const double &t, const double &eta = 9);

//For vector valued functions
std::vector<double> phi(const std::vector<double> &t_vec, double eta = 9);

auto test_function_derivative(int radius, double dt, int order = 0);

xt::xarray<double> build_test_function_matrix(const xt::xarray<double> &tt, int radius, int order=0);

xt::xarray<double> build_full_test_function_matrix(const xt::xarray<double> &tt,const xt::xarray<int> &radii, int order=0);

std::vector<std::vector<std::size_t> > get_test_function_support_indices(const int &radius, int len_tt,
                                                                         std::optional<int> n_test_functions =
                                                                                 std::nullopt);

double find_min_radius_int_error(xt::xarray<double> &U, xt::xarray<double> &tt,
    double radius_min, double radius_max,int n_test_functions, int num_radii=100, int sub_sample_rate = 2);

size_t get_corner_index(const xt::xarray<double> &yy, const std::optional<xt::xarray<double>>& xx_in = std::nullopt);

inline std::function<double(double)>
make_scalar_function(const SymEngine::Expression& expr, const SymEngine::RCP<const SymEngine::Symbol>& var){
    return [expr, var](double x) {
        SymEngine::map_basic_basic subs;
        subs[var] = SymEngine::real_double(x);
        SymEngine::Expression substituted = expr.subs(subs);
        return SymEngine::eval_double(*substituted.get_basic());
    };
}

