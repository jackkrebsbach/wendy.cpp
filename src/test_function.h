#pragma once
#include <xtensor/containers/xarray.hpp>

//Default test function used, eta controls the shape
double phi(const double &t, const double &eta = 9);

double dphi_dt(const double &t, const double &eta);

//For vector valued functions
std::vector<double> phi(const std::vector<double> &t_vec, double eta = 9);

std::function<double(double)> test_function_derivative(const double radius, const double dt, const int order = 0);


xt::xarray<double> build_test_function_matrix(const xt::xtensor<double, 1> &tt, int radius, int order = 0);

xt::xarray<double> build_full_test_function_matrix(const xt::xtensor<double, 1> &tt, const xt::xtensor<int, 1> &radii,
                                                   int order = 0);

std::vector<std::vector<std::size_t> > get_test_function_support_indices(
    const int &radius, size_t len_tt);

std::tuple<int, xt::xarray<double>, xt::xtensor<int, 1>>find_min_radius_int_error(xt::xtensor<double, 2> &U, xt::xtensor<double, 1> &tt,double radius_min, double radius_max, int num_radii = 100, int sub_sample_rate = 2);


