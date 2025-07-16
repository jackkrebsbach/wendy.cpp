#pragma once
#include "weak_residual.h"
#include "symbolic_utils.h"
#include <xtensor/misc/xsort.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

template <typename T, typename Predicate>
int find_last(const xt::xtensor<T,1>& arr, Predicate pred) {
    for (int i = arr.size() - 1; i >= 0; --i) {
        if (pred(arr.flat(i))) return i;
    }
    return -1;
}

inline std::function<double(double)>
make_scalar_function(const SymEngine::Expression& expr, const SymEngine::RCP<const SymEngine::Symbol>& var){
    return [expr, var](const double x) {
        SymEngine::map_basic_basic subs;
        subs[var] = SymEngine::real_double(x);
        const SymEngine::Expression substituted = expr.subs(subs);
        return SymEngine::eval_double(*substituted.get_basic());
    };
}


inline f_functor build_f(const std::vector<SymEngine::Expression>& f_symbolic, const size_t D, const size_t J) {
    auto dx = build_f_visitors(f_symbolic, D, J); // Symengine object to call numerical input
    return {std::move(dx),D};

}

inline J_f_functor build_J_f(const std::vector<std::vector<SymEngine::Expression>> &J_f_symbolic, const size_t &D, const size_t &J) {
    auto dx = build_jacobian_visitors(J_f_symbolic, D, J); // Symengine object to call numerical input
    return {std::move(dx)};
}


inline size_t get_corner_index(const xt::xtensor<double, 1> &yy, const xt::xtensor<double, 1>* xx_in = nullptr) {
    auto N = yy.size();

    xt::xtensor<double,1> xx;
    if (xx_in == nullptr) {
        xx = xt::arange<double>(1, N+1);
    } else {
        xx = *xx_in;
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

        // Calculate the errors (add in small # in denominator to avoid division by zero)
        auto y1_view = xt::view(yy_scaled, xt::range(0, i+1));
        auto err1 = xt::sum(xt::abs(l1 - y1_view)/(y1_view + 1e-12));
        auto y2_view = xt::view(yy_scaled, xt::range(i, yy_scaled.size()));
        auto err2 = xt::sum(xt::abs(l2 - y2_view)/(y2_view + 1e-12));
        errors[i] = err1() + err2();
    }

    auto inf = std::numeric_limits<double>::infinity();
    auto errs = xt::where(xt::isnan(errors), inf, errors);
    auto ix = xt::argmin(errs)();
    return ix;
}
