#pragma once
#include <xtensor/containers/xarray.hpp>

double phi(const double &t, const double &eta = 9);

//For vector valued functions
std::vector<double> phi(const std::vector<double> &t_vec, double eta = 9);

xt::xarray<double> build_test_function_matrix(const xt::xarray<double> &tt, int radius);

std::vector<std::vector<std::size_t> > get_test_function_support_indices(const int &radius, int len_tt,
                                                                         std::optional<int> n_test_functions =
                                                                                 std::nullopt);

double find_min_radius_int_error(xt::xarray<double> &U, xt::xarray<double> &tt,
    double radius_min, double radius_max,int n_test_functions, int num_radii=100, int sub_sample_rate = 2);

size_t get_corner_index(const xt::xarray<double> &yy, const std::optional<xt::xarray<double>>& xx_in = std::nullopt);

