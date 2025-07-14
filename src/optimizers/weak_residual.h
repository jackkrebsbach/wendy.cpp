#ifndef WEAK_RESIDUAL_H
#define WEAK_RESIDUAL_H

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
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
    );

// r(p) = g(p) - b
// b = vec[\dot{Phi}U]
template <typename F>
xt::xtensor<double,1> r(
    std::vector<double>& p,
    xt::xtensor<double, 2>& U,
    xt::xtensor<double, 1>& tt,
    xt::xtensor<double, 2>& V,
    xt::xtensor<double, 2>& V_prime,
    F& f,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& dx
    );

#endif //WEAK_RESIDUAL_H
