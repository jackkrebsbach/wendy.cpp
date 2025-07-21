#include "mle.h"

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
    const H_f_functor &Jp_JU_f_,
    const T_f_functor &Jp_Jp_JU_f_

): L(L_), U(U_), tt(tt_), V(V_), V_prime(V_prime_), b(b_), g(g_),
   JU_g(J_g_functor(U, tt, V, Ju_f_)), Jp_g(J_g_functor(U, tt, V, Jp_f_)), Jp_JU_g(H_g_functor(U, tt, V, Jp_JU_f_)), Jp_Jp_JU_g(T_g_functor({U, tt, V, Jp_Jp_JU_f_})),
   S_inv_r(S_inv_r_functor({L, g, b})), K(V_.shape()[0]), mp1(U.shape()[0]), D(U.shape()[1]) {
   J = Jp_JU_g.grad2_len;
}

double MLE::operator()(const std::vector<double> &p) const {
    const auto Lp = L(p);
    const auto r = g(p) - b;
    const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
    const auto y = xt::linalg::solve(Lp, r); // Solve Ly = g(p) - b;
    const auto x = xt::linalg::solve(xt::transpose(Lp), y); // Solve L^T x = y
    const auto logdetS = std::log(xt::linalg::det(S));
    const auto quad = xt::linalg::dot(r, x)();
    const auto wnll = logdetS + quad;
    return (wnll);
}

xt::xtensor<double, 1> MLE::Jacobian(const std::vector<double> &p) const {
    // Precomputions
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto S = xt::linalg::dot(Lp, xt::transpose(Lp)); // S(p) (Covariance)

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {3}), {K * D, D}); // ∇ₚg(p) ∈ ℝ^(K*D x D)

    // Precomputed ∇ₚS(p) gradient of the covariance matrix, 3D Tensor
    const auto Jp_LLT = xt::transpose(xt::linalg::tensordot(Jp_Lp, xt::transpose(Lp), {1},{0}), {0,2,1}); //∇ₚLLᵀ
    const auto Jp_Sp = Jp_LLT + xt::transpose(Jp_LLT, {1, 0, 2}); // ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ 3D tensor

    // Output
    xt::xtensor<double, 1> J_wnn = xt::zeros<double>({p.size()});
    for (int i = 0; i < p.size(); ++i) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_Sp_i = xt::view(Jp_Sp, xt::all(), xt::all(), i);
        const auto Jp_gp_i = xt::view(Jp_gp, xt::all(), i);
        const auto JU_gp_i = xt::view(JU_gp, xt::all(), i);

        // Compute  ∇ₚS(p)^(-1) = -S(p)^(-1)∇ₚS(p)S(p)^(-1) from L and  ∇ₚS(p)
        const auto Y = xt::linalg::solve(S, Jp_Sp_i); // S^(-1)𝜕pᵢS(p)
        const auto Xt = xt::linalg::solve(xt::transpose(S), xt::transpose(Y));
        const auto Jp_Sp_inv = -1 * xt::transpose(Xt);

        const double prt1 = 0.5 * xt::linalg::trace(Y)();
        const double prt2 = 2 * xt::linalg::dot(xt::eval(xt::transpose(Jp_gp_i)), S_inv_rp)();
        const double prt3 = xt::linalg::dot(xt::linalg::dot(xt::transpose(r), Jp_Sp_inv), r)();

        J_wnn[i] = prt1 + prt2 + prt3;
    }

    return (J_wnn);
}

//TODO: Finish implementation
xt::xtensor<double, 2> MLE::Hessian(const std::vector<double> &p) const {
    // Precomputions
    const auto Lp = L(p); // L(p)
    const auto Jp_Lp = L.Jacobian(p); // ∇ₚL(p)
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)

    const auto r = g(p) - b; // r(p) = g(p) - b
    const auto S_inv_rp = S_inv_r(p); // S^(-1)r(p)

    // Precomputed partial information of g(p) w.r.t p⃗ and U⃗
    const auto JU_gp = xt::reshape_view(JU_g(p), {K * D, D * mp1}); // ∇ᵤg(p) ∈ ℝ^(K*D x D*mp1)
    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {3}), {K * D, D}); // ∇ₚg(p) ∈ ℝ^(K*D x D)

    // Precompute S(p) = LLᵀ
    const auto Sp = xt::linalg::dot(Lp, xt::transpose(Lp)); // S(p) (Covariance)

    // Precomputed ∇ₚS(p) gradient of the covariance matrix, 3D Tensor
    const auto Jp_LLT = xt::transpose(xt::linalg::tensordot(Jp_Lp, xt::transpose(Lp), {1},{0}), {0,2,1}); //∇ₚLLᵀ
    const auto Jp_Sp = Jp_LLT + xt::transpose(Jp_LLT, {1, 0, 2}); // ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ 3D tensor

    // Precomputed first part of HₚS(p) Hessian of the covariance matrix, 4D Tensor
    // ∇ₚ∇ₚS(p) = ∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ + (∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ)ᵀ 4D tensor where Lᵀ is broadcasted
    const auto Hp_LLT = xt::transpose(xt::linalg::tensordot(Hp_Lp, xt::transpose(Lp), {1},{0}),{0,3,1,2}); //∇ₚ∇ₚLLᵀ
    // ∇ₚL∇ₚLᵀ
    auto Jp_LpT = xt::transpose(Jp_Lp, {1, 0, 2});
    auto Jp_Lp_Jp_LpT_ = xt::linalg::tensordot(Jp_Lp, Jp_LpT, {1}, {0});
    auto Jp_Lp_Jp_LpT = xt::transpose(Jp_Lp_Jp_LpT_, {0, 2, 1, 3});

    // Output
    xt::xtensor<double, 2> H_wnn = xt::zeros<double>({p.size(), p.size()});
    for (int i = 0; i < p.size(); ++i) {
        // Extract partial information for each p_i from the gradients
        const auto Jp_Sp_i = xt::view(Jp_Sp, xt::all(), xt::all(), i);
        const auto Jp_gp_i = xt::view(Jp_gp, xt::all(), i);
        const auto JU_gp_i = xt::view(JU_gp, xt::all(), i);

        // Compute  ∇ₚS(p)^(-1) = -S(p)^(-1)∇ₚS(p)S(p)^(-1) from L and  ∇ₚS(p)
        const auto Y = xt::linalg::solve(Sp, Jp_Sp_i); // S^(-1)𝜕pᵢS(p)
        const auto Xt = xt::linalg::solve(xt::transpose(Sp), xt::transpose(Y));
        const auto Jp_Sp_inv = -1 * xt::transpose(Xt);

        const auto Jp_Lp_i = xt::view(Jp_Lp, xt::all(), xt::all(), i);
        for (int j = 0; j < p.size(); ++j) {
            // Jacobian information for 𝜕ⱼS(p)
            const auto Jp_Sp_j = xt::view(Jp_Sp, xt::all(), xt::all(), j);
            // Hessian information for each p_jp_i from the Hessian 𝜕ₚ𝜕ₚS(p)
            const auto Hp_LLT_ij = xt::view(Hp_LLT, xt::all(), xt::all(), i, j);
            const auto Jp_Lp_Jp_LpT_ij  = xt::view(Jp_Lp_Jp_LpT,xt::all(), xt::all(), i, j);
            const auto H_ij =  Hp_LLT_ij + Jp_Lp_Jp_LpT_ij;            //𝜕ₚ𝜕ₚLLᵀ + 𝜕ₚL𝜕ₚLᵀ
            const auto Hp_S_ij = H_ij  + xt::transpose(H_ij);       // 𝜕ₚ𝜕ₚS(p)

            //Compute  𝜕ₚ𝜕ₚS(p)^-1
            // S^-1 𝜕ⱼS(p)
            //prt1
            const auto S_inv_Jp_Spi = xt::linalg::solve_cholesky(Lp,Jp_Sp_j);
            const auto prt1_inv = xt::linalg::dot(S_inv_Jp_Spi,-1*Jp_Sp_inv);
            //prt Compute  ∇ₚS(p)^(-1) = -S(p)^(-1)∇ₚS(p)S(p)^(-1) from L and  ∇ₚS(p)
            const auto Y2 = xt::linalg::solve_cholesky(Lp, Hp_S_ij); // S^(-1)𝜕pᵢS(p)
            const auto Xt2 = xt::linalg::solve_cholesky(Lp, xt::transpose(Y2));
            const auto Jp_Sp_inv2 = -1 * xt::transpose(Xt2);





        }
    }
    return H_wnn;
}
