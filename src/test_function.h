#pragma once
#include <xtensor/containers/xarray.hpp>

double phi(const double &t, const double &eta = 9);

 void build_test_function_matrix(
  const int &radius,
  const xt::xarray<double> &tt,
  std::optional<int> number_test_functions = std::nullopt);
