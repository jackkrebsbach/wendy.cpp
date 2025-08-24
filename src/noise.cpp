#include "noise.h"
#include "utils.h"

#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>

xt::xtensor<double, 1> convolve1d_inner(const xt::xtensor<double, 1>& signal,
                                         const xt::xtensor<double, 1>& kernel) {
    const int signal_len = signal.size();
    const int kernel_len = kernel.size();

    int output_len = signal_len - kernel_len + 1;

    if (output_len <= 0) {
        throw std::runtime_error("Signal too short for convolution with this kernel");
    }

    xt::xtensor<double, 1> result = xt::zeros<double>({output_len});

    for (int i = 0; i < output_len; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kernel_len; ++j) {
            sum += signal(i + j) * kernel(j);
        }
        result(i) = sum;
    }

    return result;
}


xt::xtensor<double, 2> fdcoeffF(const int k, const double xbar, const xt::xtensor<double, 1> &x) {
    const auto n = x.shape()[0];
    if (k >= n) {
        throw std::runtime_error("length of x must be larger than k");
    }
    const int m = k; // change to m=n-1 if you want to compute coefficients for all
    // possible derivatives.  Then modify to output all of C.

    auto c1 = 1.0;
    auto c4 = x(0) - xbar;
    xt::xtensor<double, 2> C = xt::zeros<double>({
        static_cast<std::size_t>(n),
        static_cast<std::size_t>(m + 1)
    });

    C(0, 0) = 1;

    for (int i = 1; i < n; i++) {
        const auto mn = std::min(i, m);
        auto c2 = 1.0;
        const auto c5 = c4;
        c4 = x(i) - xbar;

        for (int j = 0; j < i; ++j) {
            const auto c3 = x(i) - x(j);
            c2 = c2 * c3;

            if (j == (i - 1)) {
                for (int s = mn; s >= 1; --s) {
                    C(i, s) = c1 * (s * C(i - 1, s - 1) - c5 * C(i - 1, s)) / c2;
                }
                C(i, 0) = -c1 * c5 * C(i - 1, 0) / c2;
            }

            for (int s = mn; s >= 1; --s) {
                C(j, s) = (c4 * C(j, s) - s * C(j, s - 1)) / c3;
            }
            C(j, 0) = c4 * C(j, 0) / c3;
        }
        c1 = c2;
    }

    return C;
}

xt::xtensor<double, 1> estimate_std(const xt::xtensor<double,2> &U, int k) {
    auto D = U.shape()[1];
    xt::xtensor<double,1> std = xt::zeros<double>({D});

    for (int d = 0; d < D; d++) {
        xt::xtensor<double ,1> f = xt::col(U, d);
        xt::xtensor<double, 1> x = xt::arange<double>(-k-2, k+3);
        auto C = fdcoeffF(k, 0, x);

        xt::xtensor<double, 1> filter = xt::eval(xt::view(C, xt::all(), C.shape()[1] - 1));
        filter = filter / xt::linalg::norm(filter, 2);

        auto convolved = convolve1d_inner(f, filter);
        auto squared = convolved * convolved;
        double mean_squared = xt::mean(squared)();

        std(d) = std::sqrt(mean_squared);

    }
 return(std);
}
