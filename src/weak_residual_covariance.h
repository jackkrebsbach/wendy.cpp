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
    xt::xtensor<double, 2> sqrt_Sigma_I_D;
    xt::xtensor<double, 2> phi_prime_I_D;

    CovarianceFactor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const xt::xtensor<double, 2> &V_prime_,
        const xt::xtensor<double, 2> &Sigma_,
        const J_f_functor &Ju_f_
    );

    xt::xtensor<double, 2> operator()(const std::vector<double> &p) const;
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

    // TODO: Investigate in using solve_triangular to take advantage of structure of L
    xt::xtensor<double, 1> operator()(const std::vector<double> &p) const {
        const auto Lp = L(p);
        const auto y = xt::linalg::solve(Lp, g(p) - b); // Solve Ly = g(p) - b;
        const auto x = xt::linalg::solve(xt::transpose(Lp), y); // Solve L^T x = y
        return (x);
    }
};

#endif //WEAK_RESIDUAL_COVARIANCE_H
