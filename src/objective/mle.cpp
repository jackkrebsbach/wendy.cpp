#include "mle.h"
#include "../utils.h"

#include <numbers>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>


constexpr double DIAG_REG = 1e-9;

MLE::MLE(
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

): L(L_), U(U_), tt(tt_), V(V_), V_prime(V_prime_), b(b_), g(g_),
    Ju_f(Ju_f_), Jp_f(Jp_f_), Ju_g(Ju_g_), Hp_f(Hp_f_), Jp_g(Jp_g_), Jp_Ju_g(Jp_Ju_g_), Jp_Jp_g(Jp_Jp_g_), Jp_Jp_Ju_g(Jp_Jp_Ju_g_),
    K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
    J = Jp_Ju_g.grad2_len;
    constant_term = K * D * std::log(2 * std::numbers::pi);
}

void MLE::S_inv(xt::xarray<double> &out_, const xt::xtensor<double,2> &S, const xt::xarray<double> &R) {
    try {
        const auto C = xt::linalg::cholesky(S);
        out_ = solve_cholesky(C, R);
    } catch (const std::exception &e) {
        out_ = xt::linalg::solve(S,R);
    }
}

double MLE::operator()(const std::vector<double> &p) const {
    const auto S = L.Cov(p);
    const auto r = g(p) - b;
    double logDetS;
    try {
        // This appears to be more unstable
        // const auto C = xt::linalg::cholesky(S);
        // const auto diagC_eval = xt::eval(xt::diag(C));
        // const auto filtered_diag = xt::filter(diagC_eval, xt::abs(diagC_eval) > 1e-14);
        // logDetS = 2.0 * xt::sum(xt::log(filtered_diag))();
        const auto [U, s, Vt] = xt::linalg::svd(S, false);
        logDetS = xt::sum(xt::log(s))();
    } catch (const std::exception &e) {
        logDetS = std::log(xt::linalg::det(S));
    }

    xt::xarray<double> quad_; S_inv(quad_, S, r);
    const auto quad = xt::linalg::dot(r, quad_)();

    const auto wnll = 0.5 * (logDetS + quad + constant_term);

    return (wnll);
}


xt::xtensor<double,2> MLE::Ju_r(const std::vector<double> &p) const {

    xt::xtensor<double, 3> Ju_F({mp1, D, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double,1> &u = xt::row(U, i);
        xt::view(Ju_F, i, xt::all(), xt::all()) = Ju_f(p, u, t);
    }

    xt::xtensor<double,4> Ju_r_ = xt::zeros<double>({K, mp1, D, D});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                for (int d2 = 0; d2 < D; ++d2)
                    Ju_r_(k, m, d1, d2) = V(k, m) * Ju_F(m, d1, d2) ;

    const auto Ju_r = xt::reshape_view(Ju_r_, {K * D, mp1*J}); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Ju_r;
}


xt::xtensor<double,2> MLE::Jp_r(const std::vector<double> &p) const {

    xt::xtensor<double, 3> Jp_F({mp1, D, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double,1> &u = xt::row(U, i);
        xt::view(Jp_F, i, xt::all(), xt::all()) = Jp_f(p, u, t);
    }

    xt::xtensor<double,4> Jp_r_ = xt::zeros<double>({K, mp1, D, J});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                    for (int j = 0; j < J; ++j)
                        Jp_r_(k, m, d1, j) = V(k, m) * Jp_F(m, d1, j) ;

    const auto Jp_r = xt::reshape_view(xt::sum(Jp_r_, {1}), {K * D, J}); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Jp_r;
}

xt::xtensor<double,3> MLE::Hp_r(const std::vector<double> &p) const {

    xt::xtensor<double, 4> Hp_F({mp1, D, J, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double,1> &u = xt::row(U, i);
        xt::view(Hp_F, i, xt::all(), xt::all(), xt::all()) = Hp_f(p, u, t);
    }

    xt::xtensor<double,5> Hp_r_ = xt::zeros<double>({K, mp1, D, J, J});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                for (int j1 = 0; j1 < J; ++j1)
                    for (int j2 = 0; j2 < J; ++j2)
                    Hp_r_(k, m, d1, j1, j2) = V(k, m) * Hp_F(m, d1, j1, j2);

    const auto Jp_r = xt::reshape_view(xt::sum(Hp_r_, {1}), {K * D, J, J}); // Hₚr(p) ∈ ℝ^(K*D x J x J)

    return Jp_r;
}


std::vector<double> MLE::Jacobian(const std::vector<double> &p) const {
    const auto S = L.Cov(p);
    const auto Lp = L(p);
    const auto Jp_Lp = L.Jacobian(p);

    const auto r = g(p) - b;

    const auto Jp_rp = Jp_r(p);

    xt::xarray<double> S_inv_rp; S_inv(S_inv_rp, S, r);

    std::vector<double> J_wnn(p.size());

    for (int j = 0; j < p.size(); ++j) {

        const xt::xtensor<double, 2> Jp_L_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        const auto Jp_ = xt::linalg::dot(Jp_L_j, xt::transpose(Lp));
        const auto Jp_S = Jp_ + xt::transpose(Jp_);

        const auto Jp_r_j = xt::view(Jp_rp, xt::all(), j);

        xt::xarray<double> _; S_inv(_, S, Jp_S);

        const double prt1 = xt::linalg::trace(_)();
        const double prt2 = 2.0 * xt::linalg::dot(xt::eval(xt::transpose(Jp_r_j)), S_inv_rp)();
        const double prt3 = -1.0 * xt::linalg::dot(xt::linalg::dot(xt::transpose(S_inv_rp), Jp_S), S_inv_rp)();

        J_wnn[j] = 0.5 * (prt1 + prt2 + prt3);
    }

    return (J_wnn);
}

std::vector<std::vector<double> > MLE::Hessian(const std::vector<double> &p) const {
    // Precomputations
    const auto Lp = L(p); // L(p)
    const auto Sp = L.Cov(p);
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    const auto r = g(p) - b;
    xt::xarray<double> S_inv_rp; S_inv(S_inv_rp, Sp, r);

    const auto Ju_rp = Ju_r(p); // ∇ᵤr(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_rp = Jp_r(p);
    const auto Hp_rp = Hp_r(p);

    std::vector<std::vector<double> > H_wnn(p.size(), std::vector<double>(p.size()));
    for (int j = 0; j < p.size(); ++j) {
        const auto Jp_LLT_j = xt::linalg::dot(xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), j)), xt::eval(xt::transpose(Lp)));
        const auto Jp_Sp_j = Jp_LLT_j + xt::transpose(Jp_LLT_j);

        const auto Jp_rp_j = xt::view(Jp_rp, xt::all(), j);
        const auto Ju_rp_j = xt::view(Ju_rp, xt::all(), j);

        // Compute  S(p)^(-1)𝜕ⱼS(p)
        xt::xarray<double> X_j; S_inv(X_j, Sp, Jp_Sp_j);
        // Compute  𝜕ⱼS(p)^(-1) = -S(p)^-1𝜕ⱼS(p)S(p)^-1
        xt::xarray<double> _; S_inv(_ , Sp, X_j);
        auto Jp_Sp_inv_j = -1.0*_;

        const auto Jp_Lp_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        for (int i = j; i < p.size(); ++i) {
            // 𝜕ᵢS(p) (Jacobian information)
            const auto Jp_LLT_i = xt::linalg::dot(xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), i)), xt::eval(xt::transpose(Lp)));
            const auto Jp_Sp_i = Jp_LLT_i + xt::transpose(Jp_LLT_i);

            // Compute  S(p)^(-1)𝜕ᵢS(p)
            xt::xarray<double> X_i; S_inv(X_i, Sp, Jp_Sp_i);

            // Compute  𝜕ᵢS(p)^(-1) = -S(p)^-1𝜕ᵢS(p)S(p)^-1
            const auto Jp_Sp_inv_i = -1.0 * xt::linalg::solve(Sp, X_i);

            // 𝜕ᵢ g(p) (Jacobian information)
            const xt::xtensor<double,1> Jp_rp_i = xt::view(Jp_rp, xt::all(), i);

            // 𝜕ᵢ L(p) (Jacobian information)
            auto Jp_Lp_i = xt::view(Jp_Lp, xt::all(), xt::all(), i);

            // 𝜕ᵢ𝜕ⱼS(p) (Hessian information)
            const auto Hp_ijL_LT = xt::linalg::dot(xt::eval(xt::view(Hp_Lp, xt::all(), xt::all(), j, i)), xt::transpose(Lp));
            const auto Jp_Lp_j_Jp_Lp_iT = xt::linalg::dot(xt::eval(Jp_Lp_j), xt::transpose(xt::eval(Jp_Lp_i)));

            const auto H_ij = Hp_ijL_LT + Jp_Lp_j_Jp_Lp_iT; //𝜕ᵢ𝜕ⱼLLᵀ + 𝜕ⱼL𝜕ᵢLᵀ
            const auto Hp_S_ij = H_ij + xt::transpose(xt::eval(H_ij)); // 𝜕ᵢ𝜕ⱼS(p)

            // 𝜕ᵢ𝜕ⱼ r(p) (Hessian information)
            const auto Hp_rp_ij = xt::view(Hp_rp, xt::all(), xt::all(), j, i);

            //𝜕ᵢ𝜕ⱼS(p)^-1
            //prt1
            const auto prt1_inv = xt::linalg::solve(Sp, xt::linalg::dot(-1.0 * Jp_Sp_inv_i, Jp_Sp_j));
            //prt2 S^(-1)𝜕ᵢ𝜕ⱼS(p)S^(-1)
            const auto prt2_inv = -1.0 * xt::linalg::solve(Sp, xt::linalg::solve(Sp, Hp_S_ij));
            // prt3
            const auto prt3_inv = xt::linalg::dot(Sp, xt::linalg::dot(Jp_Sp_j, -1.0 * Jp_Sp_inv_i));
            // adding terms together
            const auto Jp_Jp_S_inv_ij = prt1_inv + prt2_inv + prt3_inv;

            // 𝜕ᵢ𝜕ⱼ l(p) Weak negative log likelihood
            const auto x1 = xt::linalg::dot(Jp_Sp_inv_i, Jp_Sp_j);
            const auto y1 = xt::linalg::solve(Sp, Hp_S_ij);
            const auto prt1 = 0.5 * (xt::linalg::trace(x1)() + xt::linalg::trace(y1)());
            const auto prt2 = xt::linalg::dot(xt::transpose(Hp_rp_ij), S_inv_rp)();
            const auto prt3 = -1.0 * xt::linalg::dot(xt::linalg::solve(Sp, Jp_rp_j), xt::linalg::dot(Jp_Sp_i, S_inv_rp))();
            const auto prt4 = xt::linalg::dot(xt::transpose(Jp_rp_j), xt::linalg::solve(Sp, Jp_rp_i))();
            const auto prt5 = -1.0 * xt::linalg::dot(xt::linalg::dot(xt::linalg::solve(Sp,Jp_rp_i), Jp_Sp_j), S_inv_rp)();

            const auto prt6 = -0.5 * (xt::linalg::dot(xt::linalg::dot(xt::transpose(S_inv_rp), Jp_Jp_S_inv_ij), S_inv_rp)());
            const auto prt7 =  xt::linalg::dot(xt::linalg::dot(S_inv_rp, -1.0*xt::linalg::dot(Jp_Sp_i, Jp_Sp_inv_j)), r)();

            const double Hij = prt1 + prt2 + prt3 + prt4 + prt5 + prt6 + prt7;

            H_wnn[j][i] = Hij;

            if (i != j) H_wnn[i][j] = Hij; // symmetric fill
        }
    }
    return H_wnn;
}

