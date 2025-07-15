#include "../utils.h"
#include "../logger.h"
#include "weak_residual_covariance.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>


/**
 * Calculate the covariance S(p,U,t) = (∇g + ϕ'∘I)(Σ∘I)(∇gᵀ + ϕ'ᵀ∘I) = LL^T
 * We can factor Σ∘I =(Σ∘I)^1/2(Σ∘I)^1/2 because it is symmetric positive definite (it is also diagonal).
 * returns covariance S(p,U,t)
 **/
struct Covariance {
    xt::xtensor<double, 2> U;
    xt::xtensor<double, 1> tt;
    xt::xtensor<double, 2> V;
    xt::xtensor<double, 2> V_prime;
    xt::xtensor<double, 2> Sigma;
    Ju_f& Jacu_f;
    size_t D;
    size_t mp1;
    size_t K;
    xt::xtensor<double, 2> sqrt_Sigma_I_D;
    xt::xtensor<double, 2> phi_prime_I_D;

    Covariance(
        const xt::xtensor<double, 2>& U_,
        const xt::xtensor<double, 2>& tt_,
        const xt::xtensor<double, 2>& V_,
        const xt::xtensor<double, 2>& V_prime_,
        const xt::xtensor<double, 2>& Sigma_,
        Ju_f& Ju_f_
    )
    : U(U_), tt(tt_), V(V_), V_prime(V_prime_), Sigma(Sigma_),
      Jacu_f(Ju_f_), D(U_.shape()[1]), mp1(U_.shape()[1]), K(V_.shape()[0])
    {
        // Precompute Cholesky factorization of Sigma (since Sigma is diagonal)
        const auto sqrt_Sigma = xt::linalg::cholesky(Sigma);
        // (ΣxI)^1/2 same as Cholesky factorization of Σ∘I because Σ is diagonal
        sqrt_Sigma_I_D = xt::linalg::kron(sqrt_Sigma, xt::eye(D));
        // ϕ'x I_d
        phi_prime_I_D = xt::linalg::kron(V_prime, xt::eye(D));
    }

    xt::xtensor<double, 2> operator()(
        const std::vector<double>& p,
        const xt::xtensor<double, 1>& tt
    ) const {

        xt::xtensor<double, 3> Ju_F({mp1, D, D});
        for (size_t i = 0; i < mp1; ++i) {
            const double& t = tt[i];
            const auto u = xt::view(U, i, xt::all());
            auto JuFi = xt::view(Ju_F, i, xt::all(), xt::all());
            JuFi = Jacu_f(p, u, t);
        }
        // 1b
        // Create matrix with the information needed to build  ∇ᵤg
        auto V_exp = xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3);         // (K, mp1, 1, 1)
        auto Ju_F_exp = xt::expand_dims(Ju_F, 0);                                             // (1, mp1, D, D)
        auto Jug = V_exp * Ju_F_exp;                                                           // (K, mp1, D, D)
        auto Ju_g_t = xt::transpose(xt::eval(Jug), {0, 2, 1, 3});                    // (K, D, mp1, D)
        xt::xtensor<double, 2> Ju_g = xt::reshape_view(Ju_g_t, {K*D, D*mp1});                         //  ∇ᵤg ∈ ℝ^(K*D, D*mp1)


        assert(Ju_g.shape() == phi_prime_I_D.shape() && Ju_g.shape()[1] == sqrt_Sigma_I_D.shape()[0]);

        auto L = xt::linalg::dot((Ju_g + phi_prime_I_D), sqrt_Sigma_I_D);

        return xt::linalg::dot(L, xt::transpose(L));
    }
};