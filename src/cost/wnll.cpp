#include <wendy/wnll.h>
#include "../utils.h"
#include <numbers>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>

WNLL::WNLL(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const xt::xtensor<double, 2> &V_prime_,
    const Covariance &S_,
    const g_functor &g_,
    const xt::xtensor<double, 1> &b_,
    const J_f_functor &Ju_f_,
    const J_f_functor &Jp_f_,
    const H_f_functor &Hp_f_

): S(S_), U(U_), tt(tt_), V(V_), V_prime(V_prime_), b(b_), g(g_),
   Ju_f(Ju_f_), Jp_f(Jp_f_), Hp_f(Hp_f_),
   K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
    J = Jp_f.dx[0].size();
    constant_term = 0.5 * K * D * std::log(2 * std::numbers::pi);
}


// ∇ᵤr(p) ∈ ℝ^(K*D x J x J)
xt::xtensor<double, 2> WNLL::Ju_r(const std::vector<double> &p) const {
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

    const auto Ju_r = xt::eval(xt::reshape_view(Ju_r_, {K * D, mp1 * D})); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Ju_r;
}

// ∇ₚr(p) ∈ ℝ^(K*D x J x J)
xt::xtensor<double, 2> WNLL::Jp_r(const std::vector<double> &p) const {
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

    const auto Jp_r = xt::eval(xt::reshape_view(xt::sum(Jp_r_, {1}), {K * D, J})); // ∇ₚr(p) ∈ ℝ^(K*D x J)

    return Jp_r;
}

// Hₚr(p) ∈ ℝ^(K*D x J x J)
xt::xtensor<double, 3> WNLL::Hp_r(const std::vector<double> &p) const {
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

    const auto Hp_r = xt::eval(xt::reshape_view(xt::sum(Hp_r_, {1}), {K * D, J, J}));

    return Hp_r;
}


double WNLL::operator()(const std::vector<double> &p) const {
    auto Sp = S(p);

    std::unique_ptr<InverseSolver> S_inv;

    try { S_inv = std::make_unique<CholeskySolver>(Sp); } catch (...) {
        try { S_inv = std::make_unique<QRSolver>(Sp); } catch (...) {
            S_inv = std::make_unique<RegularSolve>(Sp);
        }
    }
    auto r = g(p) - b;

    double logDetS;
    try {
        // This appears to be more unstable
        // const auto C = xt::linalg::cholesky(Sp);
        // const auto diagC_eval = xt::eval(xt::diag(C));
        // const auto filtered_diag = xt::filter(diagC_eval, xt::abs(diagC_eval) > 1e-15);
        // logDetS = 2.0 * xt::sum(xt::log(filtered_diag))();
        // Computationally expensive but works for now
        const auto [_, s, __] = xt::linalg::svd(Sp, false, false);
        logDetS = xt::sum(xt::log(s))();
    } catch (...) {
        logDetS = std::log(xt::linalg::det(Sp));
    }
    const xt::xarray<double> quad_ = S_inv->solve(r);

    const auto quad = xt::linalg::dot(r, quad_)();

    const auto wnll = 0.5 * (logDetS + quad) ;

    return (wnll);
}

std::vector<double> WNLL::Jacobian(const std::vector<double> &p) const {
    auto Sp = S(p);
    auto Jp_Sp = S.Jacobian(p);
    auto Jp_rp = Jp_r(p);
    auto r = g(p) - b;

    std::unique_ptr<InverseSolver> S_inv;
    try { S_inv = std::make_unique<CholeskySolver>(Sp); } catch (...) {
        try { S_inv = std::make_unique<QRSolver>(Sp); } catch (...) {
            S_inv = std::make_unique<RegularSolve>(Sp);
        }
    }

    xt::xarray<double> S_inv_rp = S_inv->solve(r);

    std::vector<double> J_wnn(p.size());

    for (int j = 0; j < p.size(); ++j) {

        const auto Jp_S_j = xt::view(Jp_Sp, xt::all(), xt::all(), j);
        const auto Jp_r_j = xt::view(Jp_rp, xt::all(), j);

        const auto Jp_S_j_eval = xt::eval(Jp_S_j);
        const auto Jp_r_j_eval = xt::eval(Jp_r_j);

        const auto tmp = xt::eval(xt::linalg::dot(Jp_S_j_eval, S_inv_rp));
        const double prt0 = 2.0 * xt::linalg::dot(Jp_r_j_eval, S_inv_rp)();
        const double prt1 = -1.0 * xt::linalg::dot(S_inv_rp, tmp)();
        const double logDetPart =  xt::linalg::trace(S_inv->solve(Jp_S_j))();


        J_wnn[j] = 0.5 * (prt0 + prt1 + logDetPart);
    }
    return (J_wnn);
}

std::vector<std::vector<double> > WNLL::Hessian(const std::vector<double> &p) const {
    auto Sp = S(p);
    auto Jp_Sp = S.Jacobian(p);

    auto Jp_rp = Jp_r(p);
    auto Ju_rp = Ju_r(p); // ∇ᵤr(p) ∈ ℝ^(K*D x D*mp1)
    auto Hp_rp = Hp_r(p);

    auto Lp = S.L(p); // ∇ₚL(p)
    auto Jp_Lp = S.Jp_L(p); // ∇ₚL(p)
    auto Hp_Lp = S.Hp_L(p); // ∇ₚ∇ₚL(p)

    auto r = g(p) - b;

    std::unique_ptr<InverseSolver> S_inv;
    try { S_inv = std::make_unique<CholeskySolver>(Sp); } catch (...) {
        try { S_inv = std::make_unique<QRSolver>(Sp); } catch (...) {
            S_inv = std::make_unique<RegularSolve>(Sp);
        }
    }

    auto S_inv_rp = xt::eval(S_inv->solve(r));

    std::vector H_wnn(p.size(), std::vector<double>(p.size()));
    for (int j = 0; j < p.size(); ++j) {

        auto Jp_rp_j = xt::eval(xt::view(Jp_rp, xt::all(), j));
        auto Ju_rp_j = xt::eval(xt::view(Ju_rp, xt::all(), j));
        auto Jp_Sp_j = xt::eval(xt::view(Jp_Sp, xt::all(), xt::all(), j));
        auto Jp_Lp_j = xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), j));

        auto shar_ = S_inv->solve(Jp_Sp_j); // S⁻¹∂ⱼS

        for (int i = j; i < p.size(); ++i) {
            // 𝜕ᵢS(p) (Jacobian information)
            auto Jp_Sp_i = xt::eval(xt::view(Jp_Sp, xt::all(), xt::all(), i));
            auto Jp_Lp_i = xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), i)); // 𝜕ᵢL(p) (Jacobian information)
            auto Jp_rp_i = xt::eval(xt::view(Jp_rp, xt::all(), i)); // 𝜕ᵢr(p) (Jacobian information)

            auto term = xt::eval(xt::linalg::dot(Jp_Sp_i, shar_)); // ∂ᵢSS⁻¹∂ⱼS

            // 𝜕ᵢ𝜕ⱼS(p)
            auto Hp_Lp_ji = xt::eval(xt::view(Hp_Lp, xt::all(), xt::all(), j, i));
            auto p1 = xt::linalg::dot(Hp_Lp_ji,xt::transpose(Lp));
            auto p2 = xt::linalg::dot(Jp_Lp_j, xt::transpose(Jp_Lp_i));
            auto _ji = xt::eval(p1 + p2); //𝜕ᵢ𝜕ⱼLLᵀ + 𝜕ⱼL𝜕ᵢLᵀ
            auto Hp_Sp_ji = xt::eval(_ji + xt::transpose(_ji)); // 𝜕ᵢ𝜕ⱼS(p)

            auto Hp_rp_ji = xt::eval(xt::view(Hp_rp, xt::all(), j, i)); // 𝜕ᵢ𝜕ⱼ r(p)

            auto prt0 = xt::linalg::dot(xt::transpose(Hp_rp_ji), S_inv_rp)();
            auto prt1 = -1.0 * xt::linalg::dot(S_inv->solve(Jp_rp_j),xt::linalg::dot(Jp_Sp_i, S_inv_rp))();

            auto _inv_factor = xt::eval(S_inv->solve(Jp_rp_i));
            auto prt2 = xt::linalg::dot(Jp_rp_j, _inv_factor)();

            auto prt3 = -2.0 * xt::linalg::dot(_inv_factor,xt::linalg::dot( Jp_Sp_j, S_inv_rp))();
            auto prt4 = -1.0 * xt::linalg::dot(S_inv_rp, xt::linalg::dot( Hp_Sp_ji, S_inv_rp))();
            auto prt5 = 2 * xt::linalg::dot(S_inv_rp, xt::linalg::dot(term, S_inv_rp))();

            auto logDetTerm = -1.0 * xt::linalg::trace(S_inv->solve(term))() + xt::linalg::trace(
                                  S_inv->solve(Hp_Sp_ji))();

            double Hij = 0.5 * (2 * (prt0 + prt1 + prt2) + prt3 + prt4 + prt5 + logDetTerm);

            H_wnn[j][i] = Hij;

            if (i != j) H_wnn[i][j] = Hij; // symmetric fill
        }
    }
    return H_wnn;
}