#include "fft.h"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xmath.hpp>
#include <fftw3.h>
#include <vector>
#include <complex>

xt::xarray<std::complex<double> >
calculate_fft(const xt::xarray<double> &data) {
    int nrows = static_cast<int>(data.shape()[0]);
    int ncols = static_cast<int>(data.shape()[1]);
    int nfreq = ncols / 2 + 1;

    xt::xarray<std::complex<double> > f_hat = xt::zeros<std::complex<double> >({nrows, nfreq});

    std::vector<double> in(ncols);
    std::vector<std::complex<double> > out(nfreq);

    fftw_plan plan = fftw_plan_dft_r2c_1d(
        ncols, in.data(),
        reinterpret_cast<fftw_complex *>(out.data()),
        FFTW_ESTIMATE);

    for (int row = 0; row < nrows; ++row) {
        auto data_row = xt::row(data, row);
        std::copy(data_row.begin(), data_row.end(), in.begin());

        fftw_execute(plan);

        for (int freq = 0; freq < nfreq; ++freq) {
            f_hat(row, freq) = out[freq];
        }
    }

    fftw_destroy_plan(plan);
    return f_hat;
}
