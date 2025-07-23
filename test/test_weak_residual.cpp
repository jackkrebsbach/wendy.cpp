#include "./doctest.h"
#include "../src/weak_residual.h"
#include "../src/utils.h"
#include "../src/symbolic_utils.h"
#include <iostream>
#include <xtensor/containers/xtensor.hpp>
#include <symengine/parser.h>
#include <xtensor-blas/xlinalg.hpp>
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


xt::xtensor<double, 2> integrate_(
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

template<typename T>
void print_xtensor2d(const T &tensor) {
    auto shape = tensor.shape();
    if (shape.size() != 2) {
        std::cerr << "Tensor is not 2D!" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

const auto U = integrate_(p, u0, 0.0, mp1 - 1, mp1, f);
const xt::xtensor<double, 1> tt = xt::linspace<double>(0, mp1 - 1, mp1);
const xt::xtensor<double, 2> V = xt::reshape_view(xt::linspace<double>(1, K * mp1, K * mp1), {K, mp1});

const auto Ju_g = J_g_functor(U, tt, V, Ju_f);
const auto Jp_g = J_g_functor(U, tt, V, Jp_f);
const auto Jp_Ju_g = H_g_functor(U, tt, V, Jp_Ju_f);
const auto Jp_Jp_Ju_g = T_g_functor(U, tt, V, Jp_Jp_Ju_f);

TEST_CASE("f_functor takes in RealLambdaDouble Visitors Evaluation") {
    const auto out = f(p, u, t);

    REQUIRE(f.D == 2);

    const auto true_1 = p[0] - p[2] / (36 + p[1] * u[1]);
    const auto true_2 = (p[3] * u[0] - p[4]);

    CHECK(out(0) == doctest::Approx(true_1));
    CHECK(out(1) == doctest::Approx(true_2));
}

TEST_CASE("Ju_f_functor computes correct Jacobian with respect to u") {
    // Evaluate the Jacobian at (p, u, t)
    // (Assume Ju_f returns a 2x2 matrix for D=2)
    auto jac = Ju_f(p, u, t);

    // Analytical derivatives:
    // f1 = p1 - p3 / (36 + p2 * u2)
    // df1/du1 = 0
    // df1/du2 = (p3 * p2) / (36 + p2 * u2)^2

    // f2 = p4 * u1 - p5
    // df2/du1 = p4
    // df2/du2 = 0

    double denom = 36 + p[1] * u[1];
    double df1_du1 = 0.0;
    double df1_du2 = (p[2] * p[1]) / (denom * denom);
    double df2_du1 = p[3];
    double df2_du2 = 0.0;

    // Check the Jacobian values
    CHECK(jac(0,0) == doctest::Approx(df1_du1)); // df1/du1
    CHECK(jac(0,1) == doctest::Approx(df1_du2)); // df1/du2
    CHECK(jac(1,0) == doctest::Approx(df2_du1)); // df2/du1
    CHECK(jac(1,1) == doctest::Approx(df2_du2)); // df2/du2
}


TEST_CASE("Jp_Ju_f_functor computes correct Hessian with respect to p and u") {
    const auto hess = Jp_Ju_f(p, u, t);

    double denom = 36 + p[1] * u[1];
    double denom2 = denom * denom;
    double denom3 = denom2 * denom;

    // f1
    double d2f1_dp1_du1 = 0.0;
    double d2f1_dp1_du2 = 0.0;
    double d2f1_dp2_du1 = 0.0;
    double d2f1_dp2_du2 = (p[2] / denom2) - (2 * p[1] * p[2] * u[1]) / denom3;
    double d2f1_dp3_du1 = 0.0;
    double d2f1_dp3_du2 = p[1] / denom2;
    double d2f1_dp4_du1 = 0.0;
    double d2f1_dp4_du2 = 0.0;
    double d2f1_dp5_du1 = 0.0;
    double d2f1_dp5_du2 = 0.0;

    // f2
    double d2f2_dp4_du1 = 1.0;

    // Check f1
    CHECK(hess(0,0,0) == doctest::Approx(d2f1_dp1_du1));
    CHECK(hess(0,1,0) == doctest::Approx(d2f1_dp1_du2));
    CHECK(hess(0,0,1) == doctest::Approx(d2f1_dp2_du1));
    CHECK(hess(0,1,1) == doctest::Approx(d2f1_dp2_du2));
    CHECK(hess(0,0,2) == doctest::Approx(d2f1_dp3_du1));
    CHECK(hess(0,1,2) == doctest::Approx(d2f1_dp3_du2));
    CHECK(hess(0,0,3) == doctest::Approx(d2f1_dp4_du1));
    CHECK(hess(0,1,3) == doctest::Approx(d2f1_dp4_du2));
    CHECK(hess(0,0,4) == doctest::Approx(d2f1_dp5_du1));
    CHECK(hess(0,1,4) == doctest::Approx(d2f1_dp5_du2));

    // Check f2
    CHECK(hess(1,0,3) == doctest::Approx(d2f2_dp4_du1));

    // All other entries for f2 should be zero
    for (int j = 0; j < 5; ++j) {
        for (int k = 0; k < 2; ++k) {
            if (!(j == 3 && k == 0)) {
                CHECK(hess(1,j,k) == doctest::Approx(0.0));
            }
        }
    }
}

TEST_CASE("Ju_g_functor computes correct Jacobian with respect to all state variables u") {
    auto J_F = xt::xtensor<double, 3>({mp1, D, D});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::view(U, i, xt::all());
        xt::view(J_F, i, xt::all(), xt::all()) = Ju_f(p, u, t);
    }
    const xt::xtensor<double, 1> phi = xt::row(V, 0);
    const xt::xtensor<double, 1> Ju_fp_col = xt::view(J_F, xt::all(), 0, 0);

    const auto x = phi * Ju_fp_col;

    const auto Ju_gp = xt::reshape_view(Ju_g(p), {K * D, D * mp1});

    // First couple of times points
    CHECK(Ju_gp(0,0) == doctest::Approx(x[0]));
    CHECK(Ju_gp(0,1) == doctest::Approx(x[1]));
    CHECK(Ju_gp(0,2) == doctest::Approx(x[2]));
    CHECK(Ju_gp(0,3) == doctest::Approx(x[3]));

    // Last dimension and last test function
    const auto phik = xt::row(V, K - 1);

    const auto Ju_fp_D1 = xt::view(J_F, xt::all(), D - 1, 0);
    const auto Ju_fp_DD = xt::view(J_F, xt::all(), D - 1, D - 1);

    const auto y = phik * Ju_fp_D1;
    const auto yDD = phik * Ju_fp_DD;

    constexpr auto LAST_ROW = K * D - 1;

    CHECK(Ju_gp(LAST_ROW,0) == doctest::Approx(y[0]));
    CHECK(Ju_gp(LAST_ROW,1) == doctest::Approx(y[1]));
    CHECK(Ju_gp(LAST_ROW,2) == doctest::Approx(y[2]));
    CHECK(Ju_gp(LAST_ROW,3) == doctest::Approx(y[3]));


    CHECK(Ju_gp(LAST_ROW, D*mp1 - 4) == doctest::Approx(yDD[0]));
    CHECK(Ju_gp(LAST_ROW, D*mp1 - 3) == doctest::Approx(yDD[1]));
    CHECK(Ju_gp(LAST_ROW, D*mp1 - 2) == doctest::Approx(yDD[2]));
    CHECK(Ju_gp(LAST_ROW, D*mp1 - 1 ) == doctest::Approx(yDD[3]));
}

TEST_CASE("Jp_g_functor computes correct Jacobian with respect to parameters p⃗") {
    auto J_F = xt::xtensor<double, 3>({mp1, D, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::view(U, i, xt::all());
        xt::view(J_F, i, xt::all(), xt::all()) = Jp_f(p, u, t);
    }

    const xt::xtensor<double, 1> phi = xt::row(V, 0);
    const xt::xtensor<double, 1> phik = xt::row(V, K - 1);

    const double x11 = xt::sum(phi * xt::view(J_F, xt::all(), 0, 0))();
    const double xKD1 = xt::sum(phik * xt::view(J_F, xt::all(), D - 1, 0))();
    const double xKDJ = xt::sum(phik * xt::view(J_F, xt::all(), D - 1, J - 1))();
    const double xKDJm1 = xt::sum(phik * xt::view(J_F, xt::all(), D - 1, J - 2))();

    const auto Jp_gp = xt::reshape_view(xt::sum(Jp_g(p), {3}), {K * D, J}); // ∇ₚg(p) ∈ ℝ^(K*D x J)

    CHECK(Jp_gp(0,0) == doctest::Approx(x11));
    CHECK(Jp_gp(K*D-1,0) == doctest::Approx(xKD1));
    CHECK(Jp_gp(K*D-1, J-1) == doctest::Approx(xKDJ));
    CHECK(Jp_gp(K*D-1, J-2) == doctest::Approx(xKDJm1));
}


TEST_CASE("Jp_Ju_g_functor computes correct Hessian with respect to parameters p⃗ and state u⃗") {
    auto H_F = xt::xtensor<double, 4>({mp1, D, D, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::view(U, i, xt::all());
        xt::view(H_F, i, xt::all(), xt::all(), xt::all()) = Jp_Ju_f(p, u, t);
    }

    const xt::xtensor<double, 1> phi1 = xt::row(V, 0);
    const xt::xtensor<double, 1> phi2 = xt::row(V, 1);

    const auto part1 = xt::view(H_F, mp1 - 1, 0, D - 1, xt::all());
    const auto part2 = xt::view(H_F, 1, 0, 0, xt::all()); //∇p 𝜕u_21 f_1(u2)

    const auto x1 = phi1[mp1 - 1] * part1; // ∇p 𝜕u_mp1*D f_1(u_mp1) * ϕ_1mp1
    const auto x2 = phi2[1] * part2; // ∇p 𝜕u_21 f_1(u2) * ϕ_22

    const auto Jp_Ju_gp = xt::reshape_view(Jp_Ju_g(p), {K * D, mp1 * D, J}); // ∇ₚ∇ᵤg(p) ∈ ℝ^(K*D x mp1*D x J)

    CHECK(Jp_Ju_gp(0,mp1*D-1,1) == doctest::Approx(x1[1])); // 𝜕p_1 𝜕u_mp1*D f_1(u_mp1) * ϕ_1mp1
    CHECK(Jp_Ju_gp(0,mp1*D-1,2) == doctest::Approx(x1[2])); // 𝜕p_2 𝜕u_mp1*D f_1(u_mp1) * ϕ_1mp1
    CHECK(Jp_Ju_gp(1,1,3) == doctest::Approx(x2[3])); // 𝜕p_4𝜕u_21 g_2(p) = 𝜕p_4 𝜕u_21 f_1(u_2) * ϕ_22
    CHECK(Jp_Ju_gp(1,1,2) == doctest::Approx(x2[2])); // 𝜕p_3𝜕u_21 g_2(p) = 𝜕p_3 𝜕u_21 f_1(u_2) * ϕ_22
    CHECK(Jp_Ju_gp(1,1,1) == doctest::Approx(x2[1])); // 𝜕p_2𝜕u_21 g_2(p) = 𝜕p_2 𝜕u_21 f_1(u_2) * ϕ_22
    CHECK(Jp_Ju_gp(1,1,0) == doctest::Approx(x2[0])); // 𝜕p_1𝜕u_21 g_2(p) = 𝜕p_1 𝜕u_21 f_1(u_2) * ϕ_22
}

TEST_CASE("Jp_Jp_Ju_g_functor computes correct high dimensional Hessian with respect to parameters p⃗ and state u⃗") {
    auto T_F = xt::xtensor<double, 5>({mp1, D, D, J, J});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::view(U, i, xt::all());
        xt::view(T_F, i, xt::all(), xt::all(), xt::all(), xt::all()) = Jp_Jp_Ju_f(p, u, t);
    }

    const xt::xtensor<double, 1> phi1 = xt::row(V, 0);

    const auto part2 = xt::view(T_F, mp1 - 1, 0, D - 1, 1, xt::all()); //∇p 𝜕p_2 𝜕u_mp1*D f_1(u_mp1)

    const auto x = phi1[mp1 - 1] * part2; // ∇p 𝜕u_21 f_1(u2) * ϕ_1mp1

    const auto Jp_Jp_Ju_gp = xt::reshape_view(Jp_Jp_Ju_g(p), {K * D, mp1 * D, J, J});
    // ∇ₚ∇ₚ∇ᵤg(p) ∈ ℝ^(K*D x mp1*D x J x J)

    std::cout << xt::eval(xt::view(Jp_Jp_Ju_gp, 0, mp1 * D - 1, 1, xt::all())) << std::endl;

    CHECK(Jp_Jp_Ju_gp(0,mp1*D-1,1,0) == doctest::Approx(x[0])); // 𝜕p_1𝜕p_2𝜕u_mp1D g_1(p) = 𝜕p_1𝜕p_2 𝜕u_mp1D f_1(u_mp1) * ϕ_1mp1}
    CHECK(Jp_Jp_Ju_gp(0,mp1*D-1,1,1) == doctest::Approx(x[1])); // 𝜕p_2𝜕p_2𝜕u_mp1D g_1(p) = 𝜕p_2𝜕p_2 𝜕u_mp1D f_1(u_mp1) * ϕ_1mp1}
    CHECK(Jp_Jp_Ju_gp(0,mp1*D-1,1,2) == doctest::Approx(x[2])); // 𝜕p_3𝜕p_2𝜕u_mp1D g_1(p) = 𝜕p_3𝜕p_2 𝜕u_mp1D f_1(u_mp1) * ϕ_1mp1}
    CHECK(Jp_Jp_Ju_gp(0,mp1*D-1,1,3) == doctest::Approx(x[3])); // 𝜕p_4𝜕p_2𝜕u_mp1D g_1(p) = 𝜕p_4𝜕p_2 𝜕u_mp1D f_1(u_mp1) * ϕ_1mp1}
}
