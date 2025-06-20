#include <cmath>
#include "logger.h"
#include <xtensor/containers/xarray.hpp>

using namespace xt;

double phi(const double &t, const double &eta = 9) {
  return (std::exp(-eta * std::pow((1 -std::pow(t,2)), -1 )));
}

 void build_test_function_matrix(
  const int &radius,
  const xarray<double> &tt,
  std::optional<int> number_test_functions = std::nullopt) {

  const auto len_tt = tt.size();

  logger->info("Length of time {}", len_tt);
}
