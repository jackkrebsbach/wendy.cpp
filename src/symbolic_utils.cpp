#include "symbolic_utils.h"
#include <Rcpp.h>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

using namespace Rcpp;
using namespace SymEngine;

std::vector<Expression> create_symbolic_vars(const std::string &base_name,
                                             int count) {
  std::vector<Expression> vars;
  for (int i = 0; i < count; i++) {
    vars.push_back(Expression(symbol(base_name + std::to_string(i + 1))));
  }
  return vars;
}

// Inputs includes both the parameters, p1,p2,.. and the state variables u1,...
// and t
std::vector<SymEngine::Expression> create_all_symbolic_inputs(int D, int J) {
  std::vector<SymEngine::Expression> u_symbols = create_symbolic_vars("u", D);
  std::vector<SymEngine::Expression> p_symbols = create_symbolic_vars("p", J);
  SymEngine::Expression t_symbol =
      SymEngine::Expression(SymEngine::symbol("t"));

  std::vector<SymEngine::Expression> input_symbols;
  input_symbols.reserve(D + J + 1);

  for (const auto &e : u_symbols)
    input_symbols.push_back(e);
  for (const auto &e : p_symbols)
    input_symbols.push_back(e);
  input_symbols.push_back(t_symbol);

  return input_symbols;
}

std::vector<SymEngine::Expression>
create_symbolic_system(const std::vector<std::string> &f) {
  std::vector<SymEngine::Expression> dx;
  dx.reserve(f.size());
  for (const auto &s : f) {
    dx.push_back(SymEngine::parse(s));
  }
  return dx;
}

vec_basic expressions_to_vec_basic(const std::vector<Expression> &exprs) {
  vec_basic basics;
  basics.reserve(exprs.size());
  for (const auto &e : exprs)
    basics.push_back(e.get_basic());
  return basics;
}

std::vector<LambdaRealDoubleVisitor>
build_symbolic_system(const std::vector<Expression> &dx, int D, int J) {
  std::vector<Expression> input_exprs = create_all_symbolic_inputs(D, J);
  vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<LambdaRealDoubleVisitor> visitors;
  visitors.reserve(dx.size());

  for (size_t i = 0; i < dx.size(); ++i) {
    LambdaRealDoubleVisitor visitor;
    visitor.init(inputs, *dx[i].get_basic());
    visitors.push_back(std::move(visitor));
  }

  return visitors;
}
// For vector input
std::vector<std::vector<Expression>>
compute_jacobian(const std::vector<Expression> &system,
                 const std::vector<Expression> &inputs) {
  std::vector<std::vector<Expression>> jacobian(
      system.size(), std::vector<Expression>(inputs.size()));

  for (size_t i = 0; i < system.size(); ++i) {
    for (size_t j = 0; j < inputs.size(); ++j) {
      jacobian[i][j] = system[i].diff(inputs[j]);
    }
  }
  return jacobian;
}

// For matrix input
std::vector<std::vector<std::vector<Expression>>>
compute_jacobian(const std::vector<std::vector<Expression>> &matrix,
                 const std::vector<Expression> &inputs) {
  size_t rows = matrix.size();
  size_t cols = rows > 0 ? matrix[0].size() : 0;
  std::vector<std::vector<std::vector<Expression>>> jacobian(
      rows, std::vector<std::vector<Expression>>(
                cols, std::vector<Expression>(inputs.size())));

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      for (size_t k = 0; k < inputs.size(); ++k) {
        jacobian[i][j][k] = matrix[i][j].diff(inputs[k]);
      }
    }
  }
  return jacobian;
}
