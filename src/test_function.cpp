#include <cmath>
#include <iostream>
#include "logger.h"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xnorm.hpp>


using namespace xt;

double phi(const double &t, const double &eta = 9) {
  return (std::exp(-eta * std::pow((1 -std::pow(t,2)), -1 )));
}
std::vector<double> phi(const std::vector<double>& t_vec, double eta = 9) {
    std::vector<double> result;
    result.reserve(t_vec.size());
    for (const auto& t : t_vec) {
        result.push_back(std::exp(-eta * std::pow((1 - std::pow(t, 2)), -1)));
    }
    return result;
}

std::vector<std::vector<std::size_t>> get_test_function_support_indices(const int &radius, const int len_tt,
    const std::optional<int> n_test_functions = std::nullopt) {

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
xt::xarray<double> build_test_function_matrix(const xarray<double> &tt, int radius) {
   const auto len_tt = tt.size();

    // Diameter can not be larger than the interior of the domain update the radius if it is
    auto diameter = 2*radius +1;

    if (len_tt < diameter) {
         radius = static_cast<int>((len_tt - 2) / 2);
         diameter = 2*radius +1;
    }
    // For one radius get the support indices for all phi_k (different centers)
    auto indices = get_test_function_support_indices(radius, len_tt);
    // Don't include the endpoints (support is zero)
    auto xx = xt::view(xt::linspace(-1,1, diameter+2), xt::range(1,diameter-1));

    // For a given radius, the evaluation of phi_k is the same for all k, just shifted so we only have to evaluate it once
    xt::xarray<double> v_row = xt::zeros<double>({xx.size()});
    std::ranges::transform(xx, v_row.begin(), [](const double x) { return phi(x, 9.0); });

    // Normalize
    auto v_l2 = xt::norm_l2(v_row, {});
    v_row /= v_l2;

    // Add back in zero on the endpoints
    xt::xarray<double> v_row_padded = xt::zeros<double>({v_row.size() + 2});
    xt::view(v_row_padded, xt::range(1, v_row.size() + 1)) = v_row;
    xt::xarray<double> V = xt::zeros<double>({indices.size(), len_tt});

    for (size_t i = 0; i < indices.size() - 1; i++) {
        const auto& support_indices = indices[i];
        auto n_support = support_indices.size();
        xt::view(V, i, xt::keep(support_indices)) = xt::view(v_row_padded, xt::range(0, n_support));
    }

    return V;
}


double find_min_radius_int_error(xt::xarray<double> &U, xt::xarray<double> &tt,
    double radius_min, double radius_max,int n_test_functions, int num_radii=100, int sub_sample_rate = 2) {
    // Number of data points
    auto Mp1  = U.shape()[0];
    // Dimension of the system
    auto D = U.shape()[1];
    int step = std::max(1, static_cast<int>(std::ceil((radius_max - radius_min) / static_cast<double>(num_radii))));
    const auto radii = xt::arange(radius_min, radius_max, step);
    auto errors = xt::xarray<double>::from_shape(std::vector<std::size_t>{radii.size()});


    const auto IX = static_cast<int>(std::floor((Mp1 - 1) / sub_sample_rate));

    for (int i=0; i < radii.size(); ++i) {
        auto radius = static_cast<int>(radii[i]);
        auto V_r = build_test_function_matrix(tt, radius);
        auto K = V_r.shape()[0]; // Number of test functions for a given radius

        // (K, Mp1, D) Essentially element wise for each dimension phi(t_i) and u(t_i) for all phi_k
        auto G = xt::expand_dims(V_r,2) * xt::expand_dims(U, 0);

        // (K, D, Mp1) Need this so reshaping works the way we want
        auto GT = xt::xarray<double>(xt::transpose(xt::xarray<double>(G), std::vector<std::size_t>{0,2,1}));

        //For column i (index time), one row is phi_k(t_i)u(t_i)[D] <- Dth dimension
        GT.reshape({K*D, Mp1});
    }


    return 2;
}

