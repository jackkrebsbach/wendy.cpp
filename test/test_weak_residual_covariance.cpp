#include "./doctest.h"
#include "../include/wendy/weak_residual_covariance.h"
#include "../src/utils.h"
#include "../src/symbolic_utils.h"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <string>
#include <vector>

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

const auto Jp_Jp_Ju_f_symbolic = build_symbolic_jacobian(Jp_Ju_f_symbolic, p_symbolic);

const auto f = build_f(f_symbolic, D, J);
const auto Ju_f = build_J_f(Ju_f_symbolic, D, J);
const auto Jp_f = build_J_f(Jp_f_symbolic, D, J);
const auto Jp_Ju_f = build_H_f(Jp_Ju_f_symbolic, D, J);
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

const xt::xtensor<double, 2> Sigma = xt::diag(xt::linspace(1,4, D));

const auto Ju_g = J_g_functor(U, tt, V, Ju_f);
const auto Jp_g = J_g_functor(U, tt, V, Jp_f);
const auto Jp_Ju_g = H_g_functor(U, tt, V, Jp_Ju_f);
const auto Jp_Jp_Ju_g = T_g_functor(U, tt, V, Jp_Jp_Ju_f);

const auto L = Covariance(U, tt, V, V, Sigma, Ju_g, Jp_Ju_g, Jp_Jp_Ju_g);

TEST_CASE("L is (∇ᵤg(p) + I_D ⊙ V') (Σ ⊙ I_mp1)") {
    const auto Lp = L(p);

    xt::xtensor<double, 2> Lp_first_bock = xt::view(Lp, xt::range(0, K), xt::range(0, mp1));
    xt::xtensor<double, 2> Lp_second_bock = xt::view(Lp, xt::range(0, K), xt::range(mp1, 2 * mp1));

    const auto Ju_gp = xt::reshape_view(Ju_g(p), {K * D, D * mp1});

    xt::xarray<double> sqrt_diag = xt::diag(Sigma);

    xt::xtensor<double, 2> L_manual_first_bock =
            xt::eval(xt::view(Ju_gp, xt::range(0, K), xt::range(0, mp1)) + V) * sqrt_diag(0);
    xt::xtensor<double, 2> L_manual_second_bock =
            xt::eval(xt::view(Ju_gp, xt::range(0, K), xt::range(mp1, 2 * mp1))) * sqrt_diag(1);

    CHECK(xt::allclose(Lp_first_bock, L_manual_first_bock));
    CHECK(xt::allclose(Lp_second_bock, L_manual_second_bock));
}

TEST_CASE("∇ₚL: ∇ₚS(p) = ∇ₚLLᵀ + (∇ₚLLᵀ)ᵀ ") {
    const auto Lp = L(p);
    const auto Jp_Lp = L.Jacobian(p);
    const auto Jp_JUgp = xt::reshape_view(Jp_Ju_g(p), {D * K, D * mp1, J});

    for (int i = 0; i < J; ++i) {
        const auto t2 = xt::view(Jp_JUgp, xt::all(), xt::all(), i);

        const xt::xtensor<double, 2> Jp_i_manual = xt::eval(xt::linalg::dot(t2, xt::linalg::kron(Sigma, xt::eye(mp1))));
        const xt::xtensor<double, 2> Jp_i = xt::eval(xt::view(Jp_Lp, xt::all(), xt::all(), i));

        CHECK(xt::allclose(Jp_i_manual, Jp_i));
    }
}

TEST_CASE("∇ₚ∇ₚL") {
    const auto Lp = L(p);
    const auto Hp_Lp = L.Hessian(p); // ∇ₚ∇ₚL(p)
    const auto Jp_Jp_JUgp = xt::reshape_view(Jp_Jp_Ju_g(p), {D * K, D * mp1, J, J});

    for (int i = 0; i < J; ++i) {
        for (int j = 0; j < J; ++j) {
            const auto t2 = xt::view(Jp_Jp_JUgp, xt::all(), xt::all(), i, j);

            const xt::xtensor<double, 2> Hp_ji_manual = xt::eval(xt::linalg::dot(t2, xt::linalg::kron(Sigma, xt::eye(mp1))));
            const xt::xtensor<double, 2> Hp_L_ji = xt::eval(xt::view(Hp_Lp, xt::all(), xt::all(), i, j));

            CHECK(xt::allclose(Hp_ji_manual, Hp_L_ji));
        }
    }
}

TEST_CASE("S_inv_r"){

    const auto F = F_functor(f, U, tt);
    const auto L = Covariance(U, tt, V, V, Sigma, Ju_g, Jp_Ju_g, Jp_Jp_Ju_g);
    const auto g = g_functor(F, V);
    const auto b = xt::eval(-1*xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(V, U)));
    const auto S_inv_r = S_inv_r_functor(L, g, b);

    const auto Lp = L(p);
    const auto S = xt::linalg::dot(Lp, xt::transpose(Lp));
    const auto S_inv_rp = S_inv_r(p);

    CHECK(xt::allclose(S_inv_rp, xt::linalg::dot( xt::linalg::inv(S), g(p) - b )));
    }



TEST_CASE("Expand matrix") {

    const xt::xtensor<double, 2> V = xt::reshape_view(xt::linspace<double>(1, K * mp1, K * mp1), {K, mp1});
    const auto _ = xt::broadcast(xt::expand_dims(V, 2), {K, mp1,J});

    for(int i = 0; i < J; ++i) {
        CHECK(xt::allclose(V, xt::view(_, xt::all(), xt::all(), i)));
    }
}



TEST_CASE("Matrix Solve"){

    const auto Lp = L(p);
    xt::xtensor<double,2> S = xt::linalg::dot(Lp, xt::transpose(Lp));
    S = 0.5 * (S + xt::transpose(S));

    xt::xarray<double> A = xt::random::randn<double>(S.shape());

    const auto B = xt::linalg::solve(S, A);

    CHECK(xt::allclose(A, xt::linalg::dot(S,B)));

}
