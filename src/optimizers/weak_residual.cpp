#include "weak_residual.h"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/lambda_double.h>

// g(p) = vec[Phi F(p,U,t)] column wise vectorization
template <typename F>
xt::xtensor<double, 1> g(
    std::vector<double>& p,
    xt::xtensor<double, 2>& U,
    xt::xtensor<double, 1>& tt,
    xt::xtensor<double, 2>& V,
    F& f,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& dx
    ) {

    const size_t n_points = U.shape()[0];
    const size_t D = U.shape()[1];

    xt::xtensor<double, 1> F_eval = xt::empty<double>({n_points, D});

    for (std::size_t i = 0; i < U.shape()[0]; ++i) {
        auto t = tt[i];
        auto u_t = xt::view(U, i, xt::all());
        auto f_t = xt::view(F_eval, i, xt::all());
        f_t = f(p, u_t, t, dx);
    }

    auto V_F_eval = xt::linalg::dot(V, F_eval);

    return xt::ravel<xt::layout_type::column_major>(V_F_eval);
}