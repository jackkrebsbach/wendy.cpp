#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../src/symbolic_utils.h"
#include "./doctest.h"

#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>
#include <string>
#include <vector>

// In general, it is not the best practice to unit test results from a library
// however this is more of my confirming the behavior of the symbolic library.

bool expr_equal(const SymEngine::Expression &a,
                const SymEngine::Expression &b) {
  return SymEngine::str(*a.get_basic()) == SymEngine::str(*b.get_basic());
}

TEST_CASE("create_symbolic_vars creates correct variable names") {
  auto vars = create_symbolic_vars("x", 3);
  REQUIRE(vars.size() == 3);
  CHECK(expr_equal(vars[0], Expression(symbol("x1"))));
  CHECK(expr_equal(vars[1], Expression(symbol("x2"))));
  CHECK(expr_equal(vars[2], Expression(symbol("x3"))));
}

TEST_CASE("create_all_symbolic_inputs creates u, p, t symbols") {
  int D = 2, J = 3;
  auto inputs = create_all_ode_symbolic_inputs(D, J);
  REQUIRE(inputs.size() == D + J + 1);
  CHECK(expr_equal(inputs[0], Expression(symbol("p1"))));
  CHECK(expr_equal(inputs[1], Expression(symbol("p2"))));
  CHECK(expr_equal(inputs[2], Expression(symbol("p3"))));
  CHECK(expr_equal(inputs[3], Expression(symbol("u1"))));
  CHECK(expr_equal(inputs[4], Expression(symbol("u2"))));
  CHECK(expr_equal(inputs[5], Expression(symbol("t"))));
}

TEST_CASE("create_symbolic_system parses character vector to expressions") {
  std::vector<std::string> f = {"p1 - p3 / (36 + p2 * u2)", "p4 * u1 - p5"};
  auto dx = build_symbolic_f(f);

  REQUIRE(dx.size() == 2);

  CHECK(SymEngine::str(*dx[0].get_basic()).find("p1") != std::string::npos);
  CHECK(SymEngine::str(*dx[0].get_basic()).find("p2") != std::string::npos);
  CHECK(SymEngine::str(*dx[0].get_basic()).find("p3") != std::string::npos);
  CHECK(SymEngine::str(*dx[0].get_basic()).find("u2") != std::string::npos);

  CHECK(SymEngine::str(*dx[1].get_basic()).find("p4") != std::string::npos);
  CHECK(SymEngine::str(*dx[1].get_basic()).find("u1") != std::string::npos);
  CHECK(SymEngine::str(*dx[1].get_basic()).find("p5") != std::string::npos);

}

TEST_CASE("expressions_to_vec_basic converts expressions to basics") {
  auto vars = create_symbolic_vars("y", 2);
  auto basics = expressions_to_vec_basic(vars);
  REQUIRE(basics.size() == 2);
  CHECK(is_a<SymEngine::Symbol>(*basics[0]));
  CHECK(is_a<SymEngine::Symbol>(*basics[1]));
}

void print_system(const std::vector<SymEngine::Expression> &system) {
  std::cout << "\n==================== ODE system ====================\n";
  for (size_t i = 0; i < system.size(); ++i) {
    std::cout << "f[" << i << "] = " << SymEngine::str(*system[i].get_basic())
        << std::endl;
  }
  std::cout << "===================================================\n"
      << std::endl;
}

void print_jacobian(
  const std::vector<std::vector<SymEngine::Expression> > &jac) {
  std::cout << "\n==================== Jacobian =====================\n";
  for (size_t i = 0; i < jac.size(); ++i) {
    for (size_t j = 0; j < jac[i].size(); ++j) {
      std::cout << SymEngine::str(*jac[i][j].get_basic()) << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "===================================================\n"
      << std::endl;
}

TEST_CASE("Jacobian of a linear system with respect to all variables") {
  auto u = create_symbolic_vars("u", 2);
  auto p = create_symbolic_vars("p", 2);
  std::vector<SymEngine::Expression> system = {
    u[0] * p[0] + u[1] * p[1],
    u[0] * u[1]
  };
  std::vector<SymEngine::Expression> inputs = {u[0], u[1], p[0], p[1]};
  auto jac = build_symbolic_jacobian(system, inputs);

  // print_system(system);
  // print_jacobian(jac);

  REQUIRE(jac.size() == 2);
  REQUIRE(jac[0].size() == 4);
  CHECK(expr_equal(jac[0][0], p[0]));
  CHECK(expr_equal(jac[0][1], p[1]));
  CHECK(expr_equal(jac[0][2], u[0]));
  CHECK(expr_equal(jac[0][3], u[1]));
  CHECK(expr_equal(jac[1][0], u[1]));
  CHECK(expr_equal(jac[1][1], u[0]));
  CHECK(expr_equal(jac[1][2], SymEngine::Expression(0)));
  CHECK(expr_equal(jac[1][3], SymEngine::Expression(0)));
}

TEST_CASE("Jacobian of a nonlinear system (sin, cos, exp) with respect to all "
  "variables") {
  auto u = create_symbolic_vars("u", 2);
  auto p = create_symbolic_vars("p", 2);
  using SymEngine::cos;
  using SymEngine::exp;
  using SymEngine::sin;

  std::vector<SymEngine::Expression> system = {
    SymEngine::add(sin(u[0]), exp(p[0] * u[1])), cos(u[1]) * p[1]
  };
  std::vector<SymEngine::Expression> inputs = {u[0], u[1], p[0], p[1]};
  auto jac = build_symbolic_jacobian(system, inputs);

  // print_system(system);
  // print_jacobian(jac);

  REQUIRE(jac.size() == 2);
  REQUIRE(jac[0].size() == 4);
  // d/d(u0): cos(u0)
  CHECK(expr_equal(jac[0][0], cos(u[0])));
  // d/d(u1): p0*exp(p0*u1)
  CHECK(expr_equal(jac[0][1], p[0] * exp(p[0] * u[1])));
  // d/d(p0): u1*exp(p0*u1)
  CHECK(expr_equal(jac[0][2], u[1] * exp(p[0] * u[1])));
  // d/d(p1): 0
  CHECK(expr_equal(jac[0][3], SymEngine::Expression(0)));
  // d/d(u0): 0
  CHECK(expr_equal(jac[1][0], SymEngine::Expression(0)));
  // d/d(u1): -sin(u1)*p1
  CHECK(expr_equal(jac[1][1],
    SymEngine::Expression(-1) * SymEngine::sin(u[1]) * p[1]));
  // d/d(p0): 0
  CHECK(expr_equal(jac[1][2], SymEngine::Expression(0)));
  // d/d(p1): cos(u1)
  CHECK(expr_equal(jac[1][3], cos(u[1])));
}

TEST_CASE("Jacobian with respect to only u variables") {
  auto u = create_symbolic_vars("u", 2);
  auto p = create_symbolic_vars("p", 2);
  using SymEngine::exp;
  using SymEngine::sin;

  std::vector<SymEngine::Expression> system = {
    SymEngine::add(sin(u[0]), exp(p[0] * u[1])), u[0] * u[1] + p[1]
  };
  std::vector<SymEngine::Expression> inputs = {u[0], u[1]};
  auto jac = build_symbolic_jacobian(system, inputs);

  // print_system(system);
  // print_jacobian(jac);

  REQUIRE(jac.size() == 2);
  REQUIRE(jac[0].size() == 2);
  CHECK(expr_equal(jac[0][0], cos(u[0])));
  CHECK(expr_equal(jac[0][1], p[0] * exp(p[0] * u[1])));
  CHECK(expr_equal(jac[1][0], u[1]));
  CHECK(expr_equal(jac[1][1], u[0]));
}

TEST_CASE("Jacobian with respect to only p variables") {
  auto u = create_symbolic_vars("u", 2);
  auto p = create_symbolic_vars("p", 2);
  using SymEngine::exp;
  using SymEngine::sin;

  std::vector<SymEngine::Expression> system = {
    SymEngine::Expression(
      SymEngine::add(SymEngine::sin(u[0]), SymEngine::exp(p[0] * u[1]))),

    u[0] * u[1] + p[1]
  };
  std::vector<SymEngine::Expression> inputs = {p[0], p[1]};
  auto jac = build_symbolic_jacobian(system, inputs);

  // print_system(system);
  // print_jacobian(jac);

  REQUIRE(jac.size() == 2);
  REQUIRE(jac[0].size() == 2);
  CHECK(expr_equal(jac[0][0], u[1] * exp(p[0] * u[1])));
  CHECK(expr_equal(jac[0][1], SymEngine::Expression(0)));
  CHECK(expr_equal(jac[1][0], SymEngine::Expression(0)));
  CHECK(expr_equal(jac[1][1], SymEngine::Expression(1)));
}

TEST_CASE("compute_jacobian (matrix version) computes derivatives") {
  auto u = create_symbolic_vars("u", 2);
  std::vector<std::vector<Expression> > matrix = {
    {u[0] * u[0], u[0] * u[1]},
    {u[1] * u[0], u[1] * u[1]}
  };
  std::vector<Expression> inputs = {u[0], u[1]};
  auto jac = build_symbolic_jacobian(matrix, inputs);
  REQUIRE(jac.size() == 2);
  REQUIRE(jac[0].size() == 2);
  REQUIRE(jac[0][0].size() == 2);
  // d(matrix[0][0])/d(u[0]) == 2*u[0]
  CHECK(expr_equal(jac[0][0][0], Expression(2) * u[0]));
  // d(matrix[1][1])/d(u[1]) == 2*u[1]
  CHECK(expr_equal(jac[1][1][1], Expression(2) * u[1]));
}
