#include "mle.h"
#include "../utils.h"

#include <numbers>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>


constexpr double DIAG_REG = 1e-10;

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
   Ju_f(Ju_f_), Jp_f(Jp_f_), Hp_f(Hp_f_), Ju_g(Ju_g_), Jp_g(Jp_g_), Jp_Ju_g(Jp_Ju_g_), Jp_Jp_g(Jp_Jp_g_),
   Jp_Jp_Ju_g(Jp_Jp_Ju_g_),
   K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
    J = Jp_Ju_g.grad2_len;
    constant_term = 0.5 * K * D * std::log(2 * std::numbers::pi);
}


// ∇ᵤr(p) ∈ ℝ^(K*D x J x J)
xt::xtensor<double, 2> MLE::Ju_r(const std::vector<double> &p) const {
    xt::xtensor<double, 3> Ju_F({mp1, D, D});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double, 1> &u = xt::row(U, i);
        xt::view(Ju_F, i, xt::all(), xt::all()) = Ju_f(p, u, t);
    }

    xt::xtensor<double, 4> Ju_r_ = xt::zeros<double>({K, mp1, D, D});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                for (int d2 = 0; d2 < D; ++d2)
                    Ju_r_(k, m, d1, d2) = V(k, m) * Ju_F(m, d1, d2);

    const auto Ju_r = xt::reshape_view(Ju_r_, {K * D, mp1 * D}); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Ju_r;
}


// ∇ₚr(p) ∈ ℝ^(K*D x J x J)
xt::xtensor<double, 2> MLE::Jp_r(const std::vector<double> &p) const {
    xt::xtensor<double, 3> Jp_F({mp1, D, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double, 1> &u = xt::row(U, i);
        xt::view(Jp_F, i, xt::all(), xt::all()) = Jp_f(p, u, t);
    }

    xt::xtensor<double, 4> Jp_r_ = xt::zeros<double>({K, mp1, D, J});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                for (int j = 0; j < J; ++j)
                    Jp_r_(k, m, d1, j) = V(k, m) * Jp_F(m, d1, j);

    const auto Jp_r = xt::reshape_view(xt::sum(Jp_r_, {1}), {K * D, J}); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Jp_r;
}

// Hₚr(p) ∈ ℝ^(K*D x J x J)
xt::xtensor<double, 3> MLE::Hp_r(const std::vector<double> &p) const {
    xt::xtensor<double, 4> Hp_F({mp1, D, J, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const xt::xtensor<double, 1> &u = xt::row(U, i);
        xt::view(Hp_F, i, xt::all(), xt::all(), xt::all()) = Hp_f(p, u, t);
    }

    xt::xtensor<double, 5> Hp_r_ = xt::zeros<double>({K, mp1, D, J, J});

    for (int k = 0; k < K; ++k)
        for (int m = 0; m < mp1; ++m)
            for (int d1 = 0; d1 < D; ++d1)
                for (int j1 = 0; j1 < J; ++j1)
                    for (int j2 = 0; j2 < J; ++j2)
                        Hp_r_(k, m, d1, j1, j2) = V(k, m) * Hp_F(m, d1, j1, j2);

    const auto Hp_r = xt::reshape_view(xt::sum(Hp_r_, {1}), {K * D, J, J});

    return Hp_r;
}


double MLE::operator()(const std::vector<double> &p) const {
    auto S = L.Cov(p);

    std::unique_ptr<InverseSolver> S_inv;
    try {
        S_inv = std::make_unique<CholeskySolver>(S);
    } catch (...) {
        try {
            S_inv = std::make_unique<QRSolver>(S);
        } catch (...) {
            S_inv = std::make_unique<RegularSolve>(S);
        }
    }

    auto r = g(p) - b;

    double logDetS;
    try {
        // This appears to be more unstable
        // const auto diagC_eval = xt::eval(xt::diag(C));
        // const auto filtered_diag = xt::filter(diagC_eval, xt::abs(diagC_eval) > 1e-14);
        // logDetS = 2.0 * xt::sum(xt::log(filtered_diag))();
        // Computationally expensive but works for now
        const auto [_, s, __] = xt::linalg::svd(S, false, false);
        logDetS = xt::sum(xt::log(s))();
    } catch (const std::exception &e) {
        logDetS = std::log(xt::linalg::det(S));
    }
    const xt::xarray<double> quad_ = S_inv->solve(r);

    const auto quad = xt::linalg::dot(r, quad_)();

    const auto wnll = 0.5 * (logDetS + quad) + constant_term;

    return (wnll);
}

std::vector<double> MLE::Jacobian(const std::vector<double> &p) const {
    auto S = L.Cov(p);

    std::unique_ptr<InverseSolver> S_inv;
    try {
        S_inv = std::make_unique<CholeskySolver>(S);
    } catch (...) {
        try {
            S_inv = std::make_unique<QRSolver>(S);
        } catch (...) {
            S_inv = std::make_unique<RegularSolve>(S);
        }
    }

    const auto Lp = L(p);
    const auto Jp_Lp = L.Jacobian(p);

    const auto r = g(p) - b;

    const auto Jp_rp = Jp_r(p);

    xt::xarray<double> S_inv_rp = S_inv->solve(r);

    std::vector<double> J_wnn(p.size());

    for (int j = 0; j < p.size(); ++j) {
        const xt::xtensor<double, 2> Jp_L_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        const auto Jp_ = xt::eval(xt::linalg::dot(Jp_L_j, xt::transpose(Lp)));
        const auto Jp_S = Jp_ + xt::transpose(Jp_);

        const auto Jp_r_j = xt::view(Jp_rp, xt::all(), j);

        const double prt1 = xt::linalg::trace(S_inv->solve(Jp_S))();
        const double prt2 = 2.0 * xt::linalg::dot(xt::eval(xt::transpose(Jp_r_j)), S_inv_rp)();
        const double prt3 = -1.0 * xt::linalg::dot(xt::linalg::dot(xt::transpose(S_inv_rp), Jp_S), S_inv_rp)();

        J_wnn[j] = 0.5 * (prt1 + prt2 + prt3);
    }

    return (J_wnn);
}

std::vector<std::vector<double> > MLE::Hessian(const std::vector<double> &p) const {
    const auto Lp = L(p);
    const auto Sp = L.Cov(p);

    std::unique_ptr<InverseSolver> S_inv;
    try {
        S_inv = std::make_unique<CholeskySolver>(Sp);
    } catch (...) {
        try {
            S_inv = std::make_unique<QRSolver>(Sp);
        } catch (...) {
            S_inv = std::make_unique<RegularSolve>(Sp);
        }
    }

    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    xt::xtensor<double, 1> r = g(p) - b;
    xt::xarray<double> S_inv_rp = xt::eval(S_inv->solve(r));

    const auto Ju_rp = Ju_r(p); // ∇ᵤr(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_rp = Jp_r(p);
    const auto Hp_rp = Hp_r(p);

    std::vector H_wnn(p.size(), std::vector<double>(p.size()));
    for (int j = 0; j < p.size(); ++j) {
        const xt::xtensor<double, 2> Jp_L_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);
        const auto Jp_j = xt::eval(xt::linalg::dot(Jp_L_j, xt::transpose(Lp)));
        const auto Jp_Sp_j = Jp_j + xt::transpose(Jp_j);


        const xt::xtensor<double, 1> Jp_rp_j = xt::view(Jp_rp, xt::all(), j);
        const xt::xtensor<double, 1> Ju_rp_j = xt::view(Ju_rp, xt::all(), j);
        const xt::xtensor<double, 2> Jp_Lp_j = xt::view(Jp_Lp, xt::all(), xt::all(), j);

        const auto shar_ = S_inv->solve(Jp_Sp_j); // S⁻¹∂ⱼS

        for (int i = j; i < p.size(); ++i) {
            // 𝜕ᵢS(p) (Jacobian information)
            const xt::xtensor<double, 2> Jp_L_i = xt::view(Jp_Lp, xt::all(), xt::all(), i);
            const auto Jp_i = xt::eval(xt::linalg::dot(Jp_L_j, xt::transpose(Lp)));
            const auto Jp_Sp_i = Jp_i + xt::transpose(Jp_i);


            const xt::xtensor<double, 1> Jp_rp_i = xt::view(Jp_rp, xt::all(), i); // 𝜕ᵢg(p) (Jacobian information)

            auto Jp_Lp_i = xt::view(Jp_Lp, xt::all(), xt::all(), i); // 𝜕ᵢL(p) (Jacobian information)

            const auto term = xt::linalg::dot(Jp_Sp_i, shar_); // ∂ᵢSS⁻¹∂ⱼS

            // 𝜕ᵢ𝜕ⱼS(p)
            const auto Hp_ijL_LT = xt::linalg::dot(xt::eval(xt::view(Hp_Lp, xt::all(), xt::all(), j, i)),
                                                   xt::transpose(Lp));
            const auto Jp_Lp_j_Jp_Lp_iT = xt::linalg::dot(xt::eval(Jp_Lp_j), xt::transpose(xt::eval(Jp_Lp_i)));
            const auto H_ij = xt::eval(Hp_ijL_LT + Jp_Lp_j_Jp_Lp_iT); //𝜕ᵢ𝜕ⱼLLᵀ + 𝜕ⱼL𝜕ᵢLᵀ
            const auto ij_Hp_S = xt::eval(H_ij + xt::transpose(H_ij)); // 𝜕ᵢ𝜕ⱼS(p)

            xt::xtensor<double, 1> Hp_rp_ij = xt::view(Hp_rp, xt::all(), j, i); // 𝜕ᵢ𝜕ⱼ r(p)

            const auto prt0 = xt::linalg::dot(xt::transpose(Hp_rp_ij), S_inv_rp)();
            const auto prt1 = -1.0 * xt::linalg::dot(xt::transpose(S_inv->solve(Jp_rp_j)),
                                                     xt::linalg::dot(Jp_Sp_i, S_inv_rp))();
            const auto _ = xt::eval(S_inv->solve(Jp_rp_i));
            const auto prt2 = xt::linalg::dot(Jp_rp_j, _)();
            const auto prt3 = -2.0 * xt::linalg::dot(xt::linalg::dot(xt::transpose(_), Jp_Sp_j), S_inv_rp)();
            const auto prt4 = -1.0 * (xt::linalg::dot(xt::linalg::dot(xt::transpose(S_inv_rp), ij_Hp_S), S_inv_rp)());
            const auto prt5 = 2 * xt::linalg::dot(transpose(S_inv_rp), xt::linalg::dot(term, S_inv_rp))();

            const auto logDetTerm = -1.0 * xt::linalg::trace(S_inv->solve(term))() + xt::linalg::trace(
                                        S_inv->solve(ij_Hp_S))();

            const double Hij = 0.5 * (2 * (prt0 + prt1 + prt2) + prt3 + prt4 + prt5 + logDetTerm);

            H_wnn[j][i] = Hij;

            if (i != j) H_wnn[i][j] = Hij; // symmetric fill
        }
    }
    return H_wnn;
}
