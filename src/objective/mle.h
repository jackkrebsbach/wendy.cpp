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
    const J_f_functor &Ju_f;
    const J_f_functor &Jp_f;
    const H_f_functor &Hp_f;
    const J_g_functor &Ju_g;
    const J_g_functor &Jp_g;
    const H_g_functor &Jp_Ju_g;
    const H_g_functor &Jp_Jp_g;
    const T_g_functor &Jp_Jp_Ju_g;
    size_t K;
    size_t mp1;
    size_t D;
    size_t J;

    double constant_term;

    MLE(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const xt::xtensor<double, 2> &V_prime_,
        const CovarianceFactor &L_,
        const g_functor &g_,
        const xt::xtensor<double, 1> &b_,
        const J_f_functor &Ju_f_,
        const J_f_functor &Jp_f_,
        const H_f_functor &Hp_f_,
        const J_g_functor &Ju_g_,
        const J_g_functor &Jp_g_,
        const H_g_functor &Jp_Ju_g_,
        const H_g_functor &Jp_Jp_g_,
        const T_g_functor &Jp_Jp_Ju_g_
        );

   double operator()(const std::vector<double> &p) const;

   std::vector<double> Jacobian(const std::vector<double> &p) const;

    xt::xtensor<double ,2> Jp_r(const std::vector<double> &p) const;

    xt::xtensor<double ,2> Ju_r(const std::vector<double> &p) const;

    xt::xtensor<double, 3> Hp_r(const std::vector<double> &p) const;

    static void S_inv(xt::xarray<double> &out_, const xt::xtensor<double,2> &S, const xt::xarray<double> &R);

   std::vector<std::vector<double>> Hessian(const std::vector<double> &p) const;

};

#endif //MLE_H
