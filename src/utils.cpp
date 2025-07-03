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

xt::xarray<double> eigen_matrix_to_xtensor(const Eigen::MatrixXd& arr) {
    std::vector<std::size_t> shape = {static_cast<std::size_t>(arr.rows()), static_cast<std::size_t>(arr.cols())};
    xt::xarray<double> result = xt::zeros<double>(shape);

    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            result(i, j) = arr(i, j);
        }
    }
    return result;
}



