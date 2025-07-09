#include "test_function.h"
#include "fft.h"
#include "logger.h"
#include <xtensor/views/xview.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/misc/xcomplex.hpp>
#include <xtensor/core/xvectorize.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

using namespace xt;

double phi(const double &t, const double &eta) {
  return (std::exp(-eta * std::pow((1 -std::pow(t,2)), -1 )));
}

std::vector<double> phi(const std::vector<double>& t_vec, double eta) {
    std::vector<double> result;
    result.reserve(t_vec.size());
    for (const auto& t : t_vec) {
        result.push_back(std::exp(-eta * std::pow((1 - std::pow(t, 2)), -1)));
    }
    return result;
}

auto test_function_derivative(const int radius, const double dt, const int order) {
    const auto scale_factor = std::pow(radius*dt, -1*order); // Chain rule to account for (t/a)^2 we get factors of (1/a), a=dt*radius
    const SymEngine::RCP<const SymEngine::Symbol> t = SymEngine::symbol("t");
    const SymEngine::Expression expression = SymEngine::exp(-9 * SymEngine::pow((1 -SymEngine::pow(SymEngine::Expression(t),2 )), -1 ));
    const auto derivative = SymEngine::expand(scale_factor*expression.diff(t));
    return(make_scalar_function(derivative, t));
}


std::vector<std::vector<std::size_t>> get_test_function_support_indices(const int &radius, const int len_tt,
    const std::optional<int> n_test_functions) {

    const int diameter =  2 * radius + 1;
    const int len_interior = len_tt - 2;

    int n; // number of test functions actually used
    int gap; // gap between test functions
    if (diameter > len_interior) {
        throw std::invalid_argument("diameter must be less than len_interior");
    }
    if (n_test_functions == std::nullopt) {
        gap = 1; // Pack as many test functions as we can
         n = ((len_interior - diameter) / gap) +1;

    } else {
        n = n_test_functions.value();
        if (n < 1) {
            throw std::invalid_argument("n_test_functions must be positive");
        }
        if (n == 1) {
             gap = (len_interior - diameter) / 2;
        } else {
            const int max_start = len_interior - diameter;
             gap = max_start / (n - 1);
        }
    }

   std::vector<std::vector<std::size_t>> indices_list;

    for (int i = 0; i < n ; ++i) {
        auto start = i*gap+1; // Offset by 1 to skip the boundary
        auto end = start + diameter;

        if (n == 1) {
            start = gap;
            end = start + diameter;
        }

        if (end > len_tt -1) {
            start = len_tt - 1 - diameter;
            end = len_tt -1;
        }
        std::vector<std::size_t> indices_k(end - start);
        std::iota(indices_k.begin(), indices_k.end(), static_cast<std::size_t>(start));
        indices_list.emplace_back(indices_k);
    }
    return indices_list;
}

// Builds a test function matrix for one radius value.
xt::xarray<double> build_test_function_matrix(const xtensor<double,1> &tt, int radius, int order) {
   const auto len_tt = tt.size();
   const double dt = std::accumulate(std::next(tt.begin()), tt.end(), 0.0,
    [it=tt.begin()](const double sum, const double val) mutable { const double diff = val - *it++; return sum + diff; }) / (tt.size() - 1);


    // Diameter can not be larger than the interior of the domain update the radius if it is
    auto diameter = 2*radius +1;

    if (len_tt < diameter) {
         radius = static_cast<int>((len_tt - 2) / 2);
         diameter = 2*radius +1;
    }
    // For one radius get the support indices for all phi_k (different centers)
    auto indices = get_test_function_support_indices(radius, len_tt);

    // Don't include the endpoints (support is zero)
    auto lin = xt::linspace(-1.0, 1.0, diameter + 2);
    auto xx = xt::xarray<double>(xt::view(lin, xt::range(1, diameter + 1)));

    auto f = [order,radius,dt](const double t) -> double {
        if (order == 0) {
            return(phi(t, 9));
        }
        const std::function<double(double)> phi_deriv = test_function_derivative(radius,dt,order);
        return(phi_deriv(t));
    };
    // For a given radius, the evaluation of phi_k is the same for all k, just shifted so we only have to evaluate it once
    auto phi_vec = xt::vectorize(f);
    auto v_row = xt::eval(phi_vec(xx));
    v_row /= xt::norm_l2(v_row)();

    // Add back in zero on the endpoints
    xt::xtensor<double,1> v_row_padded = xt::zeros<double>({v_row.size() + 2});
    xt::view(v_row_padded, xt::range(1, v_row.size() + 1)) = v_row;
    xt::xtensor<double,2> V = xt::zeros<double>({indices.size(), len_tt});

    for (size_t i = 0; i < indices.size() - 1; i++) {
        const auto& support_indices = indices[i];
        auto n_support = support_indices.size();
        xt::view(V, i, xt::keep(support_indices)) = xt::view(v_row_padded, xt::range(0, n_support));
    }

    return V;
}

xt::xarray<double> build_full_test_function_matrix(const xt::xtensor<double,1> &tt, const xt::xtensor<int,1> &radii, const int order) {

    // Vector containing test matrices for one radius
   std::vector<xt::xtensor<double,2>> test_matrices;
   for (int i = 0; i < radii.shape()[0]; ++i) {
        // Build the test matrix for one radius
        xt::xtensor<double,2> V_k = build_test_function_matrix(tt, radii[i], order);
        test_matrices.emplace_back(std::move(V_k));
    }
    xt::xtensor<double,2> V_full = test_matrices[0];

    for (size_t i = 1; i < test_matrices.size(); ++i) {
        //Subtle bug: Must wrap concatenation in xt::xarray<double>(...) otherwise we get data loss
        V_full = xt::xtensor<double,2>(xt::concatenate(xt::xtuple(V_full, test_matrices[i]),0));
    }

    return V_full;
}


double find_min_radius_int_error(xt::xtensor<double,2> &U, xt::xtensor<double, 1> &tt,
    double radius_min, double radius_max,int n_test_functions, int num_radii, int sub_sample_rate) {
    auto Mp1  = U.shape()[0]; // Number of data points
    auto D = U.shape()[1]; // Dimension of the system

    int step = std::max(1, static_cast<int>(std::ceil((radius_max - radius_min) / static_cast<double>(num_radii))));
    const auto radii = xt::arange(radius_min, radius_max, step);
    xt::xarray<double> errors = xt::zeros<double>({radii.size()});

    const auto IX = static_cast<int>(std::floor((Mp1 - 1) / sub_sample_rate));

    for (int i=0; i < radii.size(); ++i) {
        auto radius = static_cast<int>(radii[i]);
        auto V_r = build_test_function_matrix(tt, radius);
        auto K = V_r.shape()[0]; // Number of test functions for a given radius

        // (K, Mp1, D)  Element wise for each dimension phi(t_i) and u(t_i) for all phi_k
        auto G = xt::expand_dims(V_r,2) * xt::expand_dims(U, 0);
        // (K, D, Mp1) Need this so reshaping works the way we want
        auto GT = xt::xarray<double>(xt::transpose(xt::xarray<double>(G), std::vector<std::size_t>{0,2,1}));
        //For column i (index time), one row is phi_k(t_i)u(t_i)[D] <- Dth dimension
        auto GT_reshaped = xt::reshape_view(GT, {K*D, Mp1});

        //Fast Fourier Transform
        auto f_hat_G = calculate_fft(GT_reshaped);
        auto f_hat_G_imag = xt::eval(xt::imag(xt::col(f_hat_G, IX)));
        errors[i] = xt::norm_l2(f_hat_G_imag)(); // Have to actually evaluate the expression ()
    }

    auto ix = get_corner_index(errors);
    return ix;
}

size_t get_corner_index(const xt::xtensor<double, 1> &yy, const xt::xtensor<double, 1>* xx_in) {
    auto N = yy.size();

    xt::xtensor<double,1> xx;
    if (xx_in == nullptr) {
        xx = xt::arange<double>(1, N+1);
    }

    // Scale in hopes of improving stability
    auto yy_scaled = (yy/ xt::amax(xt::abs(yy))) * N;

    xt::xtensor<double,1> errors = xt::zeros<double>({N});

    for (int i=0; i < N; ++i) {
        //Check for zeros (should never happen though)
        if (xx[i] == xx[0] || xx(xx.size() -1) == xx[i] ) {
            errors[i] = std::numeric_limits<double>::infinity();
            continue;
        }

        // First secant line
        auto slope1 = (yy_scaled[i] - yy_scaled[0]) / (xx[i] - xx[0]);
        auto l1 = slope1 * (xt::view(xx, xt::range(0, i + 1)) - xx[i]) + yy_scaled[i];

        // Second secant line
        auto slope2 = (yy_scaled[yy_scaled.size() - 1] - yy_scaled[i]) / (xx[xx.size() - 1] - xx[i]);
        auto l2 = slope2 * (xt::view(xx, xt::range(i, xx.size())) - xx[i]) + yy_scaled[i];

        // Calculate the errors
        auto y1_view = xt::view(yy_scaled, xt::range(0, i+1));
        auto err1 = xt::sum(xt::abs(l1 - y1_view)/y1_view);
        auto y2_view = xt::view(yy_scaled, xt::range(i, yy_scaled.size()));
        auto err2 = xt::sum(xt::abs(l2 - y2_view)/y2_view);
        errors[i] = err1() + err2();
    }

    auto inf = std::numeric_limits<double>::infinity();
    auto errs = xt::where(xt::isnan(errors), inf, errors);
    auto ix = xt::argmax(errs)();
    return ix;
}
