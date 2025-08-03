#ifndef MLE_H
#define MLE_H

#include "../utils.h"
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

    struct InverseSolver {
        virtual ~InverseSolver() = default;
        virtual xt::xarray<double> solve(const xt::xarray<double>& b) const = 0;
    };

    struct CholeskySolver final : InverseSolver {
        xt::xarray<double> L;

        explicit CholeskySolver(const xt::xarray<double>& S) {
            L = xt::linalg::cholesky(S);
        }

        xt::xarray<double> solve(const xt::xarray<double>& b) const override {
            return solve_cholesky(L, b);
        }
    };

    struct QRSolver final : InverseSolver {
        QRFactor F;

        explicit QRSolver(const xt::xarray<double>& S) {
            F = qr_factor(S);
        }

        xt::xarray<double> solve(const xt::xarray<double>& b) const override {
            return solve_qr(F, b);
        }
    };

    struct RegularSolve final : InverseSolver {
        xt::xarray<double> F;
        explicit RegularSolve(const xt::xarray<double>& S) {
            F = S;
        }
        xt::xarray<double> solve(const xt::xarray<double>& b) const override {
            return xt::linalg::solve(F, b);
        }
    };


   double operator()(const std::vector<double> &p) const;

   std::vector<double> Jacobian(const std::vector<double> &p) const;

    xt::xtensor<double ,2> Jp_r(const std::vector<double> &p) const;

    xt::xtensor<double ,2> Ju_r(const std::vector<double> &p) const;

    xt::xtensor<double, 3> Hp_r(const std::vector<double> &p) const;

   std::vector<std::vector<double>> Hessian(const std::vector<double> &p) const;

};

#endif //MLE_H
