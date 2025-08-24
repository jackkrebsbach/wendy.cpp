#ifndef NOISE_H
#define NOISE_H

#include <xtensor/views/xview.hpp>

xt::xtensor<double, 1> convolve1d_inner(const xt::xtensor<double, 1>& signal, const xt::xtensor<double, 1>& kernel) ;

xt::xtensor<double ,2> fdcoeffF(int k, double xbar, const xt::xtensor<double, 1> &x);

xt::xtensor<double ,1> estimate_std(const xt::xtensor<double,2> &U, int k = 6);

#endif //NOISE_H

