#include "test_function.h"
#include "fft.h"
#include "utils.h"
#include <xtensor/views/xview.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/misc/xcomplex.hpp>
#include <xtensor/core/xvectorize.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/expression.h>
#include <xtensor/core/xmath.hpp>
#include <symengine/lambda_double.h>

using namespace xt;

double phi(const double &t, const double &eta) {
    return (std::exp(-1.0*eta/(1 - t*t)));
}

double dphi_dt(const double &t, const double &eta) {
    double denom = 1 - t * t;
    double phi_val = std::exp(-eta / denom);
    return phi_val * (-eta * 2 * t) / (denom * denom);
}


std::vector<double> phi(const xt::xtensor<double,1> &t_vec, double eta) {
    std::vector<double> result;

    result.reserve(t_vec.size());

    for (const auto &t: t_vec) {
        result.push_back(std::exp(-eta * std::pow((1 - std::pow(t, 2.0)), -1.0)));
    }
    return result;
}


std::function<double(double)> test_function_derivative(const double radius, const double dt, const int order) {
    const auto scale_factor = std::pow((radius*dt), -1.0 * order); // Chain rule to account for (t/a)^2 we get factors of (1/a), a=dt*radius
    const auto t = SymEngine::symbol("t");
    const SymEngine::Expression expression = SymEngine::exp( -9.0 * SymEngine::pow((1.0 - SymEngine::pow(SymEngine::Expression(t), 2)), -1));
    const auto derivative = scale_factor * expression.diff(t);
    return (make_scalar_function(derivative, t));
}

std::vector<std::vector<std::size_t> > get_test_function_support_indices(const int &radius, const size_t len_tt) {

    const int diameter = 2 * radius + 1;
    const int len_interior = static_cast<int>(len_tt) - 2;

    const int n = len_interior - diameter + 1; // number of test functions

    if (diameter > len_interior) {
        throw std::invalid_argument("diameter must be less than len_interior");
    }

    std::vector<std::vector<std::size_t> > indices_list;

    for (int i = 1; i < n + 1; ++i) {
        const auto start = i ;
        const auto end = start + diameter;
        std::vector<std::size_t> indices_k(end - start);
        std::iota(indices_k.begin(), indices_k.end(), static_cast<std::size_t>(start));
        indices_list.emplace_back(indices_k);
    }
    return indices_list;
}

// Builds a test function matrix for one radius value.
xt::xarray<double> build_test_function_matrix(const xtensor<double, 1> &tt, int radius, int order) {
    const auto len_tt = tt.size();
    const double dt = xt::mean(xt::diff(tt))();

    // Diameter can not be larger than the interior of the domain update the radius if it is
    auto diameter = 2 * radius + 1;

    if (len_tt < diameter) {
        std::cout <<  "Warning: diameter outside of domain " << std::endl;
        radius = static_cast<int>((len_tt - 2) / 2);
        diameter = 2 * radius + 1;
    }

    // For one radius get the support indices for all phi_k (different centers)
    const auto indices = get_test_function_support_indices(radius, len_tt);

    // Don't include the endpoints (support is zero)
    auto lin = xt::linspace(-1.0, 1.0, diameter);
    auto xx = xt::xarray<double>(xt::view(lin, xt::range(1, diameter-1)));

    auto f = [order,radius,dt](const double t) -> double {
        if (order == 0) {
            return (phi(t, 9));
        }

        return dphi_dt(t, 9) * std::pow(dt*radius, -1);

        // const std::function<double(double)> phi_deriv = test_function_derivative(radius , dt, order);
        // return (phi_deriv(t));
    };
    // For a given radius, the evaluation of phi_k is the same for all k, just shifted so we only have to evaluate it once
    auto phi_vec = xt::vectorize(f);

    auto v_row = xt::eval(phi_vec(xx));

    v_row /= xt::norm_l2(xt::vectorize([](const double t){return (phi(t, 9));})(xx))();

    // Add back in zero on the endpoints
    xt::xtensor<double, 1> v_row_padded = xt::zeros<double>({v_row.size() + 2});
    xt::view(v_row_padded, xt::range(1, v_row.size()+1)) = v_row;

    xt::xtensor<double, 2> V = xt::zeros<double>({indices.size(), len_tt});

    for (size_t i = 0; i < indices.size(); i++) {
        const auto &support_indices = indices[i];
        auto n_support = support_indices.size();
        xt::view(V, i, xt::keep(support_indices)) = xt::view(v_row_padded, xt::range(0, n_support));
    }
    return V;
}

xt::xarray<double> build_full_test_function_matrix(const xt::xtensor<double, 1> &tt, const xt::xtensor<int, 1> &radii, const int order) {
    std::vector<xt::xtensor<double, 2> > test_matrices(radii.shape()[0]); // Vector containing test matrices for one radius

    for (int i = 0; i < radii.shape()[0]; ++i) {
        test_matrices[i] = build_test_function_matrix(tt, radii[i], order);  // Build the test matrix for one radius
    }
    xt::xtensor<double, 2> V_full = test_matrices[0];

    for (size_t i = 1; i < test_matrices.size(); ++i) {
        V_full = xt::xtensor<double, 2>(xt::concatenate(xt::xtuple(V_full, test_matrices[i]), 0));
    }

    return V_full;
}

std::tuple<int, xt::xarray<double>, xt::xtensor<int, 1>> find_min_radius_int_error(xtensor<double, 2> &U, xtensor<double, 1> &tt, double radius_min, double radius_max, int num_radii, int sub_sample_rate) {
    auto Mp1 = U.shape()[0]; // Number of data points
    auto D = U.shape()[1]; // Dimension of the system

    int step = std::max(1, static_cast<int>(std::ceil((radius_max - radius_min) / static_cast<double>(num_radii))));
    const xtensor<int, 1> radii = xt::arange(radius_min, radius_max, step);
    xarray<double> errors = xt::zeros<double>({radii.size()});

    const auto IX = static_cast<int>(std::floor((Mp1 - 1) / sub_sample_rate));

    for (int i = 0; i < radii.size(); ++i) {
        auto radius = static_cast<int>(radii[i]);
        auto V_r = build_test_function_matrix(tt, radius);
        auto K = V_r.shape()[0]; // Number of test functions for a given radius

        // (K, Mp1, D)  Element wise for each dimension phi(t_i) and u(t_i) for all phi_k
        auto G = xt::expand_dims(V_r, 2) * xt::expand_dims(U, 0);
        // (K, D, Mp1) Need this so reshaping works the way we want
        auto GT = xt::xarray<double>(xt::transpose(xt::xarray<double>(G), std::vector<std::size_t>{0, 2, 1}));
        //For column i (index time), one row is phi_k(t_i)u(t_i)[D] <- Dth dimension
        auto GT_reshaped = xt::reshape_view(GT, {K * D, Mp1});

        //Fast Fourier Transform
        auto f_hat_G = calculate_fft(GT_reshaped);
        auto f_hat_G_imag = xt::eval(xt::imag(xt::col(f_hat_G, IX)));
        errors[i] = xt::norm_l2(f_hat_G_imag)();
    }

    const xtensor<double, 1> radii_dbl = xt::cast<double>(radii);
    auto ix = get_corner_index(xt::log(errors), &radii_dbl);
    return std::make_tuple(ix, errors, radii);
}