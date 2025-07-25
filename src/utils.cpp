#include "utils.h"
#include "symbolic_utils.h"
#include "weak_residual.h"

#include <xtensor/misc/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

// The solve_cholesky and solve_triangular from xt is busted as of now
// https://github.com/xtensor-stack/xtensor-blas/issues/242
xt::xarray<double> solve_cholesky(const xt::xarray<double>& L, const xt::xarray<double>& B) {
    using namespace cxxlapack;

    const int n = static_cast<int>(L.shape()[0]);
    const int nrhs = static_cast<int>(B.shape()[1]);

    xt::xarray<double, xt::layout_type::column_major> A = L;
    xt::xarray<double, xt::layout_type::column_major> X = B;

    const int info = cxxlapack::potrs(
        'L',        // Lower triangle
        n,          // Matrix size
        nrhs,       // Number of RHS columns
        A.data(),   // Cholesky factor
        n,          // Leading dimension of A
        X.data(),   // RHS, overwritten with solution
        n           // Leading dimension of B/X
    );

    if (info != 0) {
        throw std::runtime_error("Cholesky solve failed with info = " + std::to_string(info));
    }

    return X;
}

std::function<double(double)>
make_scalar_function(const SymEngine::Expression &expr, const SymEngine::RCP<const SymEngine::Symbol> &var) {
    return [expr, var](const double x) {
        SymEngine::map_basic_basic subs;
        subs[var] = SymEngine::real_double(x);
        const SymEngine::Expression substituted = expr.subs(subs);
        return SymEngine::eval_double(*substituted.get_basic());
    };
}

f_functor build_f(const std::vector<SymEngine::Expression> &f_symbolic, const size_t D, const size_t J) {
    const auto dx = build_f_visitors(f_symbolic, D, J); // Symengine object to call numerical input
    return {dx, D};
}

J_f_functor build_J_f(const std::vector<std::vector<SymEngine::Expression> > &J_f_symbolic, const size_t D,
                      const size_t J) {
    const auto dx = build_jacobian_visitors(J_f_symbolic, D, J); // Symengine object to call numerical input
    return J_f_functor(dx);
}

H_f_functor build_H_f(const std::vector<std::vector<std::vector<SymEngine::Expression> > > &H_f_symbolic,
                      const size_t D,
                      const size_t J) {
    const auto dx = build_jacobian_visitors(H_f_symbolic, D, J); // Symengine object to call numerical input
    return H_f_functor(dx);
}

T_f_functor build_T_f(const std::vector<std::vector<std::vector<std::vector<SymEngine::Expression> > > > &T_f_symbolic,
                      const size_t D,
                      const size_t J) {
    const auto dx = build_jacobian_visitors(T_f_symbolic, D, J); // Symengine object to call numerical input
    return T_f_functor(dx);
}

size_t get_corner_index(const xt::xtensor<double, 1> &yy, const xt::xtensor<double, 1> *xx_in) {
    auto N = yy.size();

    xt::xtensor<double, 1> xx;
    if (xx_in == nullptr) {
        xx = xt::arange<double>(1, N + 1);
    } else {
        xx = *xx_in;
    }

    // Scale in hopes of improving stability
    auto yy_scaled = (yy / xt::amax(xt::abs(yy))) * N;

    xt::xtensor<double, 1> errors = xt::zeros<double>({N});

    for (int i = 0; i < N; ++i) {
        //Check for zeros (should never happen though)
        if (xx[i] == xx[0] || xx(xx.size() - 1) == xx[i]) {
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
        auto y1_view = xt::view(yy_scaled, xt::range(0, i + 1));
        auto err1 = xt::sum(xt::abs(l1 - y1_view) / (y1_view + 1e-12));
        auto y2_view = xt::view(yy_scaled, xt::range(i, yy_scaled.size()));
        auto err2 = xt::sum(xt::abs(l2 - y2_view) / (y2_view + 1e-12));
        errors[i] = err1() + err2();
    }

    auto inf = std::numeric_limits<double>::infinity();
    auto errs = xt::where(xt::isnan(errors), inf, errors);
    auto ix = xt::argmin(errs)();
    return ix;
}

