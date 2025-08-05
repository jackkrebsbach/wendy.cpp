#ifndef WEAK_RESIDUAL_COVARIANCE_H
#define WEAK_RESIDUAL_COVARIANCE_H

#include "weak_residual.h"
#include <xtensor/containers/xtensor.hpp>

// Weak residual covariance struct is a functor with parameters of the ode: p⃗
struct Covariance {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const xt::xtensor<double, 2> &V_prime;
    const xt::xtensor<double, 1> sig;
    const J_f_functor &Ju_f;
    const J_f_functor &Jp_f;
    const H_f_functor &Jp_Ju_f;
    const T_f_functor &Jp_Jp_Ju_f;
    size_t D;
    size_t mp1;
    size_t K;
    size_t J;
    xt::xtensor<double, 2> L0;
    xt::xtensor<double, 2> Reg_I;
    const double REG = 1.0e-10;

    Covariance(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const xt::xtensor<double, 2> &V_prime_,
        const xt::xtensor<double, 1> &sig_,
        const J_f_functor &Ju_f_,
        const J_f_functor &Jp_f_,
        const H_f_functor &Jp_Ju_f_,
        const T_f_functor &Jp_Jp_Ju_f_
    );

    xt::xtensor<double, 2> L( const std::vector<double> &p) const;

    xt::xtensor<double, 3> Jp_L(const std::vector<double> &p) const;

    xt::xtensor<double, 4> Hp_L(const std::vector<double> &p) const;

    xt::xtensor<double, 2> operator()(const std::vector<double> &p) const;

    xt::xtensor<double, 3> Jacobian(const std::vector<double> &p) const;

};

#endif //WEAK_RESIDUAL_COVARIANCE_H
