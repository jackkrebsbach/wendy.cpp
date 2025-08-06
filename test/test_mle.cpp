#include "./doctest.h"
#include "../include/wendy/wnll.h"
#include "../include/wendy/weak_residual_covariance.h"
#include "../src/utils.h"
#include "../src/symbolic_utils.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <numbers>

constexpr auto J = 5;
constexpr auto D = 2;
constexpr auto K = 3;
constexpr auto mp1 = 4;


const auto u0 = xt::xtensor<double, 1>({2, 3});
const auto p = std::vector<double>({1, 2, 3, 4, 5});
const auto u = xt::xtensor<double, 1>({2, 3});
constexpr double t = 5;

const std::vector<std::string> f_string = {"p1 - p3 / (36 + p2 * u2)", "p4 * u1 - p5"};

const auto u_symbolic = create_symbolic_vars("u", 2);
const auto p_symbolic = create_symbolic_vars("p", 5);

const auto f_symbolic = build_symbolic_f(f_string);

const auto Ju_f_symbolic = build_symbolic_jacobian(f_symbolic, u_symbolic);
const auto Jp_f_symbolic = build_symbolic_jacobian(f_symbolic, p_symbolic);

const auto Jp_Ju_f_symbolic = build_symbolic_jacobian(Ju_f_symbolic, p_symbolic);

const auto Jp_Jp_f_symbolic = build_symbolic_jacobian(Jp_f_symbolic, p_symbolic);

const auto Jp_Jp_Ju_f_symbolic = build_symbolic_jacobian(Jp_Ju_f_symbolic, p_symbolic);

const auto f = build_f(f_symbolic, D, J);
const auto Ju_f = build_J_f(Ju_f_symbolic, D, J);
const auto Jp_f = build_J_f(Jp_f_symbolic, D, J);
const auto Jp_Ju_f = build_H_f(Jp_Ju_f_symbolic, D, J);
const auto Jp_Jp_f = build_H_f(Jp_Jp_f_symbolic, D, J);
const auto Jp_Jp_Ju_f = build_T_f(Jp_Jp_Ju_f_symbolic, D, J);


static xt::xtensor<double, 2> integrate_(
    const std::vector<double> &p,
    const xt::xtensor<double, 1> &u0,
    const double t0, const double t1, int npoints,
    const std::function<xt::xtensor<double, 1>(const std::vector<double> &, const xt::xtensor<double, 1> &, double)> &f_
) {
    const int dim = u0.size();
    xt::xtensor<double, 2> result = xt::zeros<double>({npoints, dim});
    const double dt = (t1 - t0) / (npoints - 1);
    xt::xtensor<double, 1> u = u0;
    double t = t0;

    for (int i = 0; i < npoints; ++i) {
        xt::row(result, i) = u;
        auto du = f_(p, u, t);
        for (int d = 0; d < dim; ++d) {
            u[d] += du[d] * dt;
        }
        t += dt;
    }
    return result;
}



const auto U = integrate_(p, u0, 0.0, mp1 - 1, mp1, f);
const xt::xtensor<double, 1> tt = xt::linspace<double>(0, mp1 - 1, mp1);
const xt::xtensor<double, 2> V = xt::reshape_view(xt::linspace<double>(1, K * mp1, K * mp1), {K, mp1});

const xt::xtensor<double, 2> Sigma = xt::diag(xt::ones<double>({D}));

const auto F = F_functor(f, U, tt);
const auto g = g_functor(F, V);
const auto Ju_g = J_g_functor(U, tt, V, Ju_f);
const auto Jp_g = J_g_functor(U, tt, V, Jp_f);
const auto Jp_Ju_g = H_g_functor(U, tt, V, Jp_Ju_f);
const auto Jp_Jp_g = H_g_functor(U, tt, V, Jp_Jp_f);
const auto Jp_Jp_Ju_g = T_g_functor(U, tt, V, Jp_Jp_Ju_f);

const auto L = Covariance(U, tt, V, V, Sigma, Ju_g, Jp_Ju_g, Jp_Jp_Ju_g);

const auto b = xt::eval(-xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(V, U)));
const auto S_inv_r = S_inv_r_functor(L, g, b);

// constexpr auto J = 5;
// constexpr auto D = 2;
// constexpr auto K = 3;
// constexpr auto mp1 = 4;

const auto mle = WNLL(U, tt, V, V, L, g, b, Ju_g, Jp_g, Jp_Ju_g, Jp_Jp_g, Jp_Jp_Ju_g, S_inv_r);

// The solve_cholesky and solve_triangular from xt is busted as of now
// https://github.com/xtensor-stack/xtensor-blas/issues/242

TEST_CASE("Cholesky solve vs regular solve") {
    constexpr int D = 30;
    // Create symmetric positive definite matrix Sp = A * Aᵀ + δI
    xt::xarray<double> A = xt::random::randn<double>({D, D});
    xt::xarray<double> Sp = xt::linalg::dot(A, xt::transpose(A)) + 1e-3 * xt::eye<double>(D);

    // Cholesky factor
    xt::xarray<double> C = xt::linalg::cholesky(Sp);
    // Create a symmetric matrix Jp_Sp_j
    xt::xarray<double> B = xt::random::randn<double>({D, D});
    xt::xarray<double> Jp_Sp_j = 0.5 * (B + xt::transpose(B));

    // Compute -S(p)^-1𝜕ⱼS(p)S(p)^-1 from L and 𝜕ⱼS(p)
    // Cholesky solve
    const xt::xtensor<double,2> LeftInverse1 = solve_cholesky(C,xt::transpose(Jp_Sp_j));
    const auto c_solve= -1*solve_cholesky(C, xt::transpose(LeftInverse1));

    // Regular Solve
    const xt::xtensor<double,2> LeftInverse2 = xt::linalg::solve(Sp,xt::transpose(Jp_Sp_j));
    const auto r_solve = -1*xt::linalg::solve(Sp, xt::transpose(LeftInverse2));

    CHECK(xt::allclose(Sp, xt::linalg::dot(C, xt::transpose(C))));
    CHECK(xt::allclose(xt::linalg::solve(Sp, B) ,  solve_cholesky(xt::linalg::cholesky(Sp), B)));
    CHECK(xt::allclose(c_solve, r_solve));


}

// TEST_CASE("Weak Negative Log Likelihood") {
//     const auto wnll = mle(p);
//
//     const auto Lp = L(p);
//     const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
//     const auto r = g(p) - b;
//     const auto x = xt::linalg::solve(S,r);
//     const auto logdet = std::log(xt::linalg::det(S));
//     const auto quad = xt::linalg::dot(r,x)();
//     const auto wnnl_manual = 0.5*(logdet + quad + K*D*std::log(2*std::numbers::pi));
//
//     CHECK(xt::isclose(wnll, wnnl_manual));
//
// }

TEST_CASE("HESSIAN Term") {
    const auto Lp = L(p);
    const auto Jp_Lp = L.Jacobian(p);
    const auto Hp_Lp = L.Hessian(p);

    const auto Hp_LLT = xt::transpose(xt::linalg::tensordot(Hp_Lp, xt::transpose(Lp), {1}, {0}), {0, 3, 1, 2});

    for (int j = 0; j < J; ++j){
        for (int i = 0; i < J; ++i) {
            const auto Hp_LLT_ij = xt::eval(xt::view(Hp_LLT, xt::all(), xt::all(), j, i));
            const auto Hp_LLT_ij_Manual = xt::eval(xt::linalg::dot(xt::eval(xt::view(Hp_Lp, xt::all(), xt::all(), j, i)) , xt::transpose(Lp) )); //∇ₚLLᵀ
            CHECK(xt::allclose(Hp_LLT_ij, Hp_LLT_ij_Manual));
        }
     }
}

TEST_CASE("Right Symmetric Inverse") {

    // Create symmetric positive definite matrix Sp = A * Aᵀ + δI
    xt::xarray<double> A = xt::random::randn<double>({D, D});
    xt::xarray<double> Sp = xt::linalg::dot(A, xt::transpose(A)) + 1e-3 * xt::eye<double>(D);
    xt::xarray<double> B = xt::random::randn<double>(Sp.shape());

    /// Solve BA^-1 = X with A symmetric
    /// B = XA => Bᵀ = AᵀXᵀ => Bᵀ = AXᵀ  solve => xᵀ

    const auto Xt = xt::linalg::solve(Sp, xt::transpose(B));
    const auto X = xt::transpose(xt::eval(Xt));

    CHECK(xt::allclose(B, xt::linalg::dot(X,Sp)));

}
