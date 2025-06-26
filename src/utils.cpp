#include "utils.h"
#include <Eigen/Dense>
#include <xtensor/containers/xarray.hpp>

Eigen::MatrixXd xtensor_matrix_to_eigen(const xt::xarray<double>& arr) {
    auto shape = arr.shape();
    Eigen::MatrixXd mat(shape[0], shape[1]);
    for (size_t i = 0; i < shape[0]; ++i)
        for (size_t j = 0; j < shape[1]; ++j)
            mat(i, j) = arr(i, j);
    return mat;
}

