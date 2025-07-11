//
// Created by Jack Krebsbach on 7/11/25.
//

#ifndef WEAK_RESIDUAL_H
#define WEAK_RESIDUAL_H

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/lambda_double.h>

xt::xarray<double> f(
    std::vector<double>& p, // parameters of the system
    const xt::xtensor<double, 1>& u, // state for one time point
    double &t,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& dx
    );

// Juf the jacobian of f with respect to the state, u.
xt::xtensor<double,2> eval_J_uf(
    std::vector<double>& p, // parameters of the system
    const xt::xtensor<double, 1>& u, // state for one time point
    const double &t, // time stamp
    std::vector<std::vector<SymEngine::LambdaRealDoubleVisitor>> & jf // symbolic representation of J_uf
    );

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
