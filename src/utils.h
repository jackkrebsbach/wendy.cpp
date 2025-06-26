#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <xtensor/containers/xarray.hpp>

Eigen::MatrixXd xtensor_matrix_to_eigen(const xt::xarray<double>& arr);

#endif //UTILS_H
