#include "./doctest.h"
#include "../include/wendy/wnll.h"
#include "../include/wendy/weak_residual_covariance.h"
#include "../src/utils.h"
#include "../src/symbolic_utils.h"
#include <iostream>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/generators/xrandom.hpp>

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
const xt::xtensor<double, 2> V_prime = 3*xt::reshape_view(xt::linspace<double>(1, K * mp1, K * mp1), {K, mp1});

const xt::xtensor<double, 2> Sigma = xt::diag(xt::ones<double>({D}));

const auto F = F_functor(f, U, tt);
const auto g = g_functor(F, V);
const auto Ju_g = J_g_functor(U, tt, V, Ju_f);
const auto Jp_g = J_g_functor(U, tt, V, Jp_f);
const auto Jp_Ju_g = H_g_functor(U, tt, V, Jp_Ju_f);
const auto Jp_Jp_g = H_g_functor(U, tt, V, Jp_Jp_f);
const auto Jp_Jp_Ju_g = T_g_functor(U, tt, V, Jp_Jp_Ju_f);

const auto L = Covariance(U, tt, V, V_prime, Sigma, Ju_g, Jp_Ju_g, Jp_Jp_Ju_g);

const auto b = xt::eval(-xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(V, U)));
const auto S_inv_r = S_inv_r_functor(L, g, b);

const auto mle = WNLL(U, tt, V, V_prime, L, g, b, Ju_g, Jp_g, Jp_Ju_g, Jp_Jp_g, Jp_Jp_Ju_g, S_inv_r);


TEST_CASE("Decomposition of SVD") {
    const auto SVD = xt::linalg::svd(V, false);
    const xt::xtensor<double, 2> U = std::get<0>(SVD);
    const xt::xtensor<double, 1> singular_values = std::get<1>(SVD);
    const xt::xtensor<double, 2> Vt = std::get<2>(SVD);

    CHECK(xt::allclose(V,xt::linalg::dot(U, xt::linalg::dot( xt::diag(singular_values), Vt))));

}