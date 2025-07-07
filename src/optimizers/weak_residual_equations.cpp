#include "weak_residual_equations.h"
#include <xtensor/containers/xarray.hpp>


template <typename F>
xt::xarray<double> g(std::vector<double>&p, xt::xarray<double>&tt, F& f) {


    return xt::xarray<double>({1,2,3});
}
