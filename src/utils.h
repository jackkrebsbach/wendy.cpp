#pragma once
#include <Eigen/Dense>
#include <xtensor/containers/xarray.hpp>


Eigen::MatrixXd xtensor_matrix_to_eigen(const xt::xarray<double>& arr);

xt::xarray<double> eigen_matrix_to_xtensor(const Eigen::MatrixXd& arr);

template <typename T, typename Predicate>
int find_last(const xt::xarray<T>& arr, Predicate pred) {
    for (int i = arr.size() - 1; i >= 0; --i) {
        if (pred(arr.flat(i))) return i;
    }
    return -1;
}

template <typename EigenVector>
xt::xarray<typename EigenVector::Scalar> eigen_to_xtensor_1d(const EigenVector& vec) {
    using Scalar = typename EigenVector::Scalar;
    // Evaluate the expression to a concrete Eigen vector
    Eigen::Array<Scalar, Eigen::Dynamic, 1> evaluated = vec.eval();
    xt::xarray<Scalar> result = xt::zeros<Scalar>({static_cast<size_t>(evaluated.size())});
    for (Eigen::Index i = 0; i < evaluated.size(); ++i) {
        result[i] = evaluated(i);
    }
    return result;
}


