#ifndef WEAK_RESIDUAL_COVARIANCE_H
#define WEAK_RESIDUAL_COVARIANCE_H

#include "weak_residual.h"
#include <xtensor/containers/xtensor.hpp>

// Weak residual covariance struct is a functor with parameters of the ode: p⃗
struct CovarianceFactor {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const xt::xtensor<double, 2> &V_prime;
    const xt::xtensor<double, 2> &Sigma;
    J_g_functor JU_g;
    size_t D;
    size_t mp1;
    size_t K;
    xt::xtensor<double, 2> sqrt_Sigma_I_D;
    xt::xtensor<double, 2> phi_prime_I_D;

    CovarianceFactor(
        const xt::xtensor<double, 2>& U_,
        const xt::xtensor<double, 1>& tt_,
        const xt::xtensor<double, 2>& V_,
        const xt::xtensor<double, 2>& V_prime_,
        const xt::xtensor<double, 2>& Sigma_,
        const J_f_functor& Ju_f_
    );

    xt::xtensor<double, 2> operator()(const std::vector<double>& p) const;
};

#endif //WEAK_RESIDUAL_COVARIANCE_H
