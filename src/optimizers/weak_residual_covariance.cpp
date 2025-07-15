#include "../utils.h"
#include "../logger.h"
#include "weak_residual_covariance.h"
#include  "weak_residual.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>


/**
 * Calculate the covariancefactor L for S(p,U,t) = (∇g + ϕ'∘I)(Σ∘I)(∇gᵀ + ϕ'ᵀ∘I) = LL^T
 * We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is symmetric positive definite (it is also diagonal).
 * callable covariance of L where LLᵀ=S(p) : the data, U and t, are constant.
 **/

CovarianceFactor::CovarianceFactor(
        const xt::xtensor<double, 2>& U_,
        const xt::xtensor<double, 2>& tt_,
        const xt::xtensor<double, 2>& V_,
        const xt::xtensor<double, 2>& V_prime_,
        const xt::xtensor<double, 2>& Sigma_,
        J_f_functor& Ju_f_
    )
    : U(U_), tt(tt_), V(V_), V_prime(V_prime_),
      Sigma(Sigma_), JU_g(JU_g_functor(U_, tt_, V_, Ju_f_)),
      D(U_.shape()[1]), mp1(U_.shape()[1]), K(V_.shape()[0])
    {
        // Precompute square root of Sigma (Sigma is diagonal)
        const auto sqrt_Sigma = xt::linalg::cholesky(Sigma);
        // (ΣxI)^1/2 same as Cholesky factorization of Σ∘I because Σ is diagonal
        sqrt_Sigma_I_D = xt::linalg::kron(sqrt_Sigma, xt::eye(D));
        // ϕ'x I_d
        phi_prime_I_D = xt::linalg::kron(V_prime, xt::eye(D));
    }

    xt::xtensor<double, 2> CovarianceFactor::operator()(
        const std::vector<double>& p
    ) const {

        auto Ju_g_eval = JU_g(p);                                                                 // gradient information for a given set of parameters p
        auto Ju_g_expanded = xt::expand_dims(Ju_g_eval, 0);                                        // (1, mp1, D, D)
        auto V_expanded = xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3);    // (K, mp1, 1, 1)
        auto Jug = V_expanded * Ju_g_expanded;                                                 // (K, mp1, D, D)
        auto Ju_g_t = xt::transpose(xt::eval(Jug), {0, 2, 1, 3});                    // (K, D, mp1, D)
        auto Ju_g = xt::reshape_view(Ju_g_t, {K*D, D*mp1});                                  //  ∇ᵤg ∈ ℝ^(K*D, D*mp1)

        assert(Ju_g.shape() == phi_prime_I_D.shape() && Ju_g.shape()[1] == sqrt_Sigma_I_D.shape()[0]);

        auto L = xt::linalg::dot((Ju_g + phi_prime_I_D), sqrt_Sigma_I_D);
        return(L);
    };