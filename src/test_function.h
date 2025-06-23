#pragma once
#include <xtensor/containers/xarray.hpp>

double phi(const double &t, const double &eta = 9);

xt::xarray<double> build_test_function_matrix(const xt::xarray<double> &tt, int radius);

std::vector<std::vector<std::size_t>> get_test_function_support_indices(const int &radius, int len_tt,
     std::optional<int> n_test_functions = std::nullopt);
