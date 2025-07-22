#include "./doctest.h"
#include "../src/weak_residual.h"
#include "../src/utils.h"
#include "../src/symbolic_utils.h"

#include <xtensor/containers/xtensor.hpp>
#include <symengine/parser.h>
#include <string>
#include <vector>

constexpr auto J =  5;
constexpr auto D =  2;

const auto p =  std::vector<double>({1,2,3,4,5});
const auto u =  xt::xtensor<double,1>({2,3});
constexpr double t = 5;

const std::vector<std::string> f_string = {"p1 - p3 / (36 + p2 * u2)", "p4 * u1 - p5"};

const auto u_symbolic =  create_symbolic_vars("u", 2);
const auto p_symbolic =  create_symbolic_vars("u", 5);

const auto f_symbolic = build_symbolic_f(f_string);

const auto Ju_f_symbolic = build_symbolic_jacobian(f_symbolic, u_symbolic);
const auto Jp_f_symbolic = build_symbolic_jacobian(f_symbolic, p_symbolic);

const auto Jp_Ju_f_symbolic = build_symbolic_jacobian(f_symbolic, p_symbolic);

const auto Jp_Jp_Ju_f_symbolic = build_symbolic_jacobian(f_symbolic, p_symbolic);

const auto f = build_f(f_symbolic, D, J);
const auto Ju_f = build_J_f(Ju_f_symbolic, D, J);

TEST_CASE("f_functor takes in RealLambdaDouble Visitors Evaluation") {

    const auto out = f(p, u ,t);

    REQUIRE(f.D == 2);

    const auto true_1 = p[0] - p[2]/(36 + p[1]*u[1]);
    const auto true_2 = (p[3]*u[0] - p[4]);

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
