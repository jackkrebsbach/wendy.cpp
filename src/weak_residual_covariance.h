#ifndef WEAK_RESIDUAL_COVARIANCE_H
#define WEAK_RESIDUAL_COVARIANCE_H

#include "weak_residual.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>

// Weak residual covariance struct is a functor with parameters of the ode: p⃗
struct CovarianceFactor {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const xt::xtensor<double, 2> &V_prime;
    const xt::xtensor<double, 2> &Sigma;
    const J_g_functor &Ju_g;
    const H_g_functor &Jp_Ju_g;
    const T_g_functor &Jp_Jp_Ju_g;
    const J_f_functor &Ju_f;
    size_t D;
    size_t mp1;
    size_t K;
    size_t J;
    xt::xtensor<double, 2> Sigma_I_mp1;
    xt::xtensor<double, 2> I_D_phi_prime;
    xt::xtensor<double, 2> L0;

    CovarianceFactor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const xt::xtensor<double, 2> &V_prime_,
        const xt::xtensor<double, 2> &Sigma_,
        const J_g_functor &Ju_g_,
        const H_g_functor &Jp_Ju_g_,
        const T_g_functor &Jp_Jp_Ju_g_,
        const J_f_functor &Ju_f_
    );

    xt::xtensor<double, 2> operator()(const std::vector<double> &p) const;

    xt::xtensor<double, 3> Jacobian(const std::vector<double> &p) const;

    xt::xtensor<double, 4> Hessian(const std::vector<double> &p) const;
};

// S(p)^-1(g(p) - b) is a function of the parameters p⃗
struct S_inv_r_functor {
    const CovarianceFactor &L;
    const g_functor &g;
    const xt::xtensor<double, 1> &b;

    S_inv_r_functor(
        const CovarianceFactor &L_,
        const g_functor &g_,
        const xt::xtensor<double, 1> &b_
    ): L(L_), g(g_), b(b_) {
    }

    xt::xtensor<double, 1> operator()(const std::vector<double> &p) const {
        const auto Lp = L(p);
        const auto S = xt::linalg::dot(Lp, xt::eval(xt::transpose(Lp)));
        const auto r = g(p) - b;
        const auto x = xt::linalg::solve(S, r); // Solve Sx = g(p) - b;
        return (x);
    }
};

#endif //WEAK_RESIDUAL_COVARIANCE_H
