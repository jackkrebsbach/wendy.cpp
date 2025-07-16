#pragma once
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <complex>

xt::xarray<std::complex<double> >
calculate_fft(const xt::xarray<double> &data);
