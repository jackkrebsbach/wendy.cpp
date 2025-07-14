#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>

using namespace SymEngine;

std::vector<Expression> create_symbolic_vars(const std::string &base_name,
                                             const size_t count) {
  std::vector<Expression> vars;
  vars.reserve(count);
for (int i = 0; i < count; i++) {
    vars.emplace_back(symbol(base_name + std::to_string(i)));
  }
  return vars;
}

// Inputs includes both the parameters, p1, p2,... and the state variables u1, ..., and t
std::vector<SymEngine::Expression> create_all_symbolic_inputs(const size_t D, const size_t J) {
  const std::vector<SymEngine::Expression> u_symbols = create_symbolic_vars("u", D);
  const std::vector<SymEngine::Expression> p_symbols = create_symbolic_vars("p", J);
  const auto t_symbol =
      SymEngine::Expression(SymEngine::symbol("t"));

  std::vector<SymEngine::Expression> input_symbols;
  input_symbols.reserve(D + J + 1);

  for (const auto &e: p_symbols)
    input_symbols.push_back(e);
  for (const auto &e: u_symbols)
    input_symbols.push_back(e);
  input_symbols.push_back(t_symbol);

  return input_symbols;
}

std::vector<SymEngine::Expression>
build_symbolic_f(const std::vector<std::string> &f) {
  std::vector<SymEngine::Expression> dx;
  dx.reserve(f.size());
  for (const auto &s: f) {
    dx.emplace_back(SymEngine::parse(s));
  }
  return dx;
}

vec_basic expressions_to_vec_basic(const std::vector<Expression>& exprs) {
  vec_basic basics;
  basics.reserve(exprs.size());
  for (const auto &e: exprs)
    basics.push_back(e.get_basic());
  return basics;
}

std::vector<std::unique_ptr<LambdaRealDoubleVisitor>>
build_f_visitors(const std::vector<Expression> &dx, const size_t D, const size_t J) {

  const std::vector<Expression> input_exprs = create_all_symbolic_inputs(D, J);
  const vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<std::unique_ptr<LambdaRealDoubleVisitor>> visitors;
  visitors.reserve(dx.size());

  for (const auto &i: dx) {
    auto visitor = std::make_unique<LambdaRealDoubleVisitor>();
    visitor->init(inputs, *i.get_basic());
    visitors.push_back(std::move(visitor));
  }

  return visitors;
}

std::vector<std::vector<std::unique_ptr<LambdaRealDoubleVisitor>>>
build_jacobian_visitors(const std::vector<std::vector<Expression> > &J_uf, const size_t D, const size_t J) {

  const std::vector<Expression> input_exprs = create_all_symbolic_inputs(D, J);
  const vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<std::vector<std::unique_ptr<LambdaRealDoubleVisitor>>> visitors;
  visitors.reserve(D);

  for (size_t i = 0; i < D; ++i) {
    std::vector<std::unique_ptr<LambdaRealDoubleVisitor>> row;
    row.reserve(D);
    for (size_t j = 0; j < D; ++j) {
        row.emplace_back(std::make_unique<LambdaRealDoubleVisitor>());
      }
    visitors.emplace_back(std::move(row));
  }

  for (size_t i = 0; i < D; ++i) {
    for (size_t j = 0; j < D; ++j) {
      auto basic  = J_uf[i][j].get_basic();
      visitors[i][j]->init(inputs, *basic);
    }
  }

  return visitors;
}

// For vector input
std::vector<std::vector<Expression> >
build_symbolic_jacobian(const std::vector<Expression> &system,
                 const std::vector<Expression> &inputs) {
  std::vector<std::vector<Expression> > jacobian(
    system.size(), std::vector<Expression>(inputs.size()));

  for (size_t i = 0; i < system.size(); ++i) {
    for (size_t j = 0; j < inputs.size(); ++j) {
      jacobian[i][j] = system[i].diff(inputs[j]);
    }
  }
  return jacobian;
}

// For matrix input
std::vector<std::vector<std::vector<Expression> > >
build_symbolic_jacobian(const std::vector<std::vector<Expression> > &matrix,
                 const std::vector<Expression> &inputs) {
  size_t rows = matrix.size();
  size_t cols = rows > 0 ? matrix[0].size() : 0;
  std::vector<std::vector<std::vector<Expression> > > jacobian(
    rows, std::vector<std::vector<Expression> >(
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
