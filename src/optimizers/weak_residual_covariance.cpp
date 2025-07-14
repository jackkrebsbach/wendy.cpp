#include "weak_residual.h"
#include "weak_residual_covariance.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <symengine/lambda_double.h>

// /**
//  * Calculate the covariance S(p,U,t) = (∇g + ϕ'∘I)(Σ∘I)(∇gᵀ + ϕ'ᵀ∘I) = LL^T
//  * We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is symmetric positive definite (it is also diagonal).
//  * returns covariance S(p,U,t)
//  **/

template <typename F>
xt::xtensor<double, 2> covariance(
    std::vector<double>  &p,
    xt::xtensor<double, 2> &U,
    xt::xtensor<double, 1> &tt,
    xt::xtensor<double, 2> &V,
    F &Ju_f // Jacobian of f w.r.t the state variables: Jᵤf(p,u,t)
    ) {

    // Build L where LL^T = S

    // 1 ∇ᵤg gradient of g with respect to the state
    // 1a Jᵤf(p,u,t) first we store all the gradient information in a 3D tensor
    const size_t D = U.shape()[1];
    const size_t mp1 = U.shape()[0];

    xt::xtensor<double, 3> J_uF({mp1, D, D});

    for (size_t i =0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto u = xt::view(U,i,xt::all());
        auto JuFi = xt::view(J_uF, i, xt::all(), xt::all());
        JuFi = Ju_f(p, u, t);
    }
    // 1b

    // ϕ'∘I

    // (Σ∘I)^1/2

    return V;
}