#ifndef MLE_H
#define MLE_H

#include "../weak_residual_covariance.h"
#include "../weak_residual.h"

struct MLE {
    const CovarianceFactor &L;
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const xt::xtensor<double, 2> &V_prime;
    const xt::xtensor<double, 1> &b;
    const g_functor &g;
    const J_g_functor &JU_g;
    const J_g_functor &Jp_g;
    const H_g_functor &Jp_JU_g;
    const H_g_functor &Jp_Jp_g;
    const T_g_functor &Jp_Jp_JU_g;
    const S_inv_r_functor &S_inv_r;
    size_t K;
    size_t mp1;
    size_t D;
    size_t J;

    MLE(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const xt::xtensor<double, 2> &V_prime_,
        const CovarianceFactor &L_,
        const g_functor &g_,
        const xt::xtensor<double, 1> &b_,
        const J_g_functor &Ju_g_,
        const J_g_functor &Jp_g_,
        const H_g_functor &Jp_JU_g_,
        const H_g_functor &Jp_Jp_g_,
        const T_g_functor &Jp_Jp_JU_g_,
        const S_inv_r_functor & S_inv_r_
        );

   double operator()(const std::vector<double> &p) const;

   std::vector<double> Jacobian(const std::vector<double> &p) const;

   std::vector<std::vector<double>> Hessian(const std::vector<double> &p) const;

};

#endif //MLE_H
