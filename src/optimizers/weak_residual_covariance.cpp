#include "../utils.h"
#include "weak_residual_covariance.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

/**
 * Calculate the covariance S(p,U,t) = (∇g + ϕ'∘I)(Σ∘I)(∇gᵀ + ϕ'ᵀ∘I) = LL^T
 * We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is symmetric positive definite (it is also diagonal).
 * returns covariance S(p,U,t)
 **/

xt::xtensor<double, 2> covariance(
    std::vector<double>  &p,
    xt::xtensor<double, 2> &U,
    xt::xtensor<double, 1> &tt,
    xt::xtensor<double, 2> &V,
    Ju_f &Ju_f // Jacobian Jᵤf(p,u,t)
    ) {

    // Build L where LL^T = S
    // 1 ∇ᵤg gradient of g with respect to the state
    // 1a Jᵤf(p,u,t) first we store all the gradient information in a 3D tensor
    const size_t D = U.shape()[1];
    const size_t mp1 = U.shape()[0];
    const size_t K = V.shape()[0];

    xt::xtensor<double, 3> Ju_F({mp1, D, D});

    for (size_t i =0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto u = xt::view(U,i,xt::all());
        auto JuFi = xt::view(Ju_F, i, xt::all(), xt::all());
        JuFi = Ju_f(p, u, t);
    }
    // 1b
    // Create matrix with the information needed to build  ∇ᵤg
    auto Vt = xt::transpose(V);
    auto Vt_exp = xt::expand_dims(xt::expand_dims(Vt, 2), 3);         // (K, mp1, 1, 1)
    auto Ju_F_exp = xt::expand_dims(Ju_F, 0);                                 // (1, mp1, D, D)
    auto Ju_g_4d = Vt_exp * Ju_F_exp;                                             // (K, mp1, D, D)
    auto Ju_g_t = xt::transpose(xt::eval(Ju_g_4d), {0, 2, 1, 3});           // (K, D, mp1, D)
    auto Ju_g_reshaped = xt::reshape_view(Ju_g_t, {K*D, D*mp1});
    xt::xtensor<double, 2> Ju_g = xt::eval(Ju_g_reshaped);                                      // (K*D, D*mp1)

    // ϕ'∘I
    // (Σ∘I)^1/2

    return V;
}