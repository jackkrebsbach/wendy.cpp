#include "utils.h"
#include "weak_residual_covariance.h"
#include "weak_residual.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/misc/xpad.hpp>
/**
 * Calculate the covariance factor L for S(p,U,t) = (∇g + ϕ'∘I)(Σ²∘I)(∇gᵀ + ϕ'ᵀ∘I) = LL^T
 * We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is symmetric positive definite (it is also diagonal).
 * callable covariance of L where LLᵀ=S(p) : the data, U and t, are constant.
 **/

CovarianceFactor::CovarianceFactor(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const xt::xtensor<double, 2> &V_prime_,
    const xt::xtensor<double, 2> &Sigma_,
    const J_g_functor &Ju_g_,
    const H_g_functor &Jp_Ju_g_,
    const T_g_functor &Jp_Jp_Ju_g_,
    const J_f_functor &Ju_f_

)
    : U(U_), tt(tt_), V(V_), V_prime(V_prime_),
      Sigma(Sigma_), Ju_g(Ju_g_), Jp_Ju_g(Jp_Ju_g_), Jp_Jp_Ju_g(Jp_Jp_Ju_g_), Ju_f(Ju_f_) {

    mp1 = U.shape()[0];
    D = U.shape()[1];
    K = V.shape()[0];
    J = Jp_Ju_g.grad2_len;

    Sigma_I_mp1 = xt::linalg::kron( xt::eye(mp1), Sigma);
    I_D_phi_prime = xt::linalg::kron(V_prime, xt::eye(D));

    xt::xtensor<double,4> L0_ = xt::zeros<double>({K, D, mp1, D});

    auto sig = xt::eval(xt::diag(Sigma));
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t m = 0; m < mp1; ++m)
            for (std::size_t d = 0; d < D; ++d)
                L0_(k, d, m, d) = V_prime(k, m) * sig(d);

    L0 = xt::xtensor<double,2>(xt::reshape_view(L0_, {K * D, mp1 * D}));
}

// L(p) where Covariance = S(p) = L(p)L(p)ᵀ
xt::xtensor<double, 2> CovarianceFactor::operator()(
    const std::vector<double> &p
) const {
    const auto sig = xt::eval(xt::diag(Sigma));
    const auto sig_view = xt::reshape_view(sig, std::vector<size_t>({1, 1, 1, D}));

    xt::xtensor<double, 3> J_F({mp1, D, D});
    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i]; const auto &u = xt::row(U,i);
        xt::view(J_F, i, xt::all(), xt::all()) = Ju_f(p, u, t);
    }

    xt::xtensor<double, 4> L1_ = xt::zeros<double>({K, D, mp1, D});

    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t d2 = 0; d2 < D; ++d2)
            for (std::size_t m = 0; m < mp1; ++m)
                for (std::size_t d1 = 0; d1 < D; ++d1)
                    L1_(k, d2, m, d1) = J_F(m, d2, d1) * V(k, m) * sig(d1);

    auto L1 = xt::reshape_view(L1_, {K * D, mp1 * D});
    const auto L = xt::eval(L1 + L0);

    return (L);
};

// ∇ₚL(p) gradient of the Covariance factor where ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ
xt::xtensor<double, 3> CovarianceFactor::Jacobian(const std::vector<double> &p) const {

    const auto sig = xt::eval(xt::diag(Sigma));  // shape (D,)
    const auto sig_kron = xt::tile(sig, mp1);

    const auto Jp_Ju_gp = xt::reshape_view(Jp_Ju_g(p), {D * K, mp1 * D, J});

    const auto& shape = Jp_Ju_gp.shape();
    size_t N = shape[0];
    size_t M = shape[1];
    size_t L = shape[2];

    xt::xtensor<double, 3> Jp_L = xt::zeros<double>({N, M, L});

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t l = 0; l < L; ++l) {
                Jp_L(i, j, l) = Jp_Ju_gp(i, j, l) * sig_kron(l);
            }
        }
    }

    return (Jp_L);
};


// ∇ₚ∇ₚL(p) Hessian of the Covariance factor where ∇ₚ∇ₚS(p) = ∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ + (∇ₚ∇ₚLLᵀ + ∇ₚL∇ₚLᵀ)ᵀ
xt::xtensor<double, 4> CovarianceFactor::Hessian(const std::vector<double> &p) const {
    const auto Jp_Jp_Ju_gp = xt::reshape_view(Jp_Jp_Ju_g(p), {D * K, D * mp1, J, J});
                                                                                                // Sigma_I_mp1 shape (D*mp1 x D*mp1)
    const auto Jp_H_ = xt::linalg::tensordot(Jp_Jp_Ju_gp, Sigma_I_mp1, {1}, {0}); // shape: (D*K,J, J, D*mp1)
    const auto Jp_H = xt::transpose(Jp_H_, {0, 3, 1, 2}); // shape: (D*K, D*mp1, J, J)

    return (Jp_H);
};
