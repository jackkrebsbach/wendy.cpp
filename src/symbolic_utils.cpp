#include "symbolic_utils.h"
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <symengine/parser.h>
#include <symengine/subs.h>

#include <wendy/wendy.h>

using namespace SymEngine;


std::vector<Expression> create_symbolic_vars(const std::string &base_name,
                                             const size_t count) {
  std::vector<Expression> vars;
  vars.reserve(count);
  for (int i = 1; i < count + 1; i++) {
    vars.emplace_back(symbol(base_name + std::to_string(i)));
  }
  return vars;
}

// Inputs includes both the parameters, p1, p2,... and the state variables u1, ..., and t
std::vector<SymEngine::Expression> create_all_ode_symbolic_inputs(const size_t D, const size_t J) {
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



vec_basic expressions_to_vec_basic(const std::vector<Expression> &exprs) {
  vec_basic basics;
  basics.reserve(exprs.size());
  for (const auto &e: exprs)
    basics.push_back(e.get_basic());
  return basics;
}

std::vector<Expression> lognormal_transform(const std::vector<Expression> &dx) {
  const size_t D = dx.size();

  const auto u_vars = create_symbolic_vars("u", D);

  // Build substitution map: uᵢ ↦ exp(uᵢ)
  map_basic_basic sub_map;
  for (const auto &xi : u_vars) {
    sub_map[xi.get_basic()] = exp(xi);
  }

  // Perform: duᵢ / uᵢ, substitute uᵢ ↦ exp(uᵢ), simplify
  std::vector<Expression> result;
  result.reserve(D);
  for (size_t i = 0; i < D; ++i) {
    Expression fud = dx[i] / u_vars[i];
    result.emplace_back(fud.subs(sub_map));
  }

  return result;
}

std::vector<SymEngine::Expression>
build_symbolic_f(const std::vector<std::string> &f, const size_t D, const NoiseDist noise_dist) {

  if (f.empty()) throw std::runtime_error("Empty symbolic expression list passed to build_symbolic_f.");

  std::vector<SymEngine::Expression> dx;
  dx.reserve(f.size());
  for (const auto &s: f) dx.emplace_back(SymEngine::parse(s));

  if (noise_dist == NoiseDist::AddGaussian) return dx;

  if (noise_dist == NoiseDist::LogNormal) {
      return lognormal_transform(dx);
  }
  return dx;
}

std::vector<std::shared_ptr<LambdaRealDoubleVisitor> >
build_f_visitors(const std::vector<Expression> &dx, const size_t D, const size_t J) {
  const std::vector<Expression> input_exprs = create_all_ode_symbolic_inputs(D, J);
  const vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > visitors;
  visitors.reserve(dx.size());

  for (const auto &i: dx) {
    auto visitor = std::make_unique<LambdaRealDoubleVisitor>();
    visitor->init(inputs, *i.get_basic());
    visitors.push_back(std::move(visitor));
  }

  return visitors;
}

// Matrix input
std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > >
build_jacobian_visitors(const std::vector<std::vector<Expression> > &J_f, const size_t D, const size_t J) {
  const size_t n_row = J_f.size();
  const size_t n_col = J_f[0].size();

  const std::vector<Expression> input_exprs = create_all_ode_symbolic_inputs(D, J);
  const vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > visitors;
  visitors.reserve(D);

  for (size_t i = 0; i < n_row; ++i) {
    std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > row;
    row.reserve(n_col);
    for (size_t j = 0; j < n_col; ++j) {
      row.emplace_back(std::make_unique<LambdaRealDoubleVisitor>());
    }
    visitors.emplace_back(std::move(row));
  }

  for (size_t i = 0; i < n_row; ++i) {
    for (size_t j = 0; j < n_col; ++j) {
      auto basic = J_f[i][j].get_basic();
      visitors[i][j]->init(inputs, *basic);
    }
  }

  return visitors;
}

// 3-D Tensor input
std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > >
build_jacobian_visitors(const std::vector<std::vector<std::vector<Expression> > > &H_f, const size_t D,
                        const size_t J) {
  const size_t n_row = H_f.size();
  const size_t n_col = n_row > 0 ? H_f[0].size() : 0;
  const size_t n_dep = (n_col > 0) ? H_f[0][0].size() : 0;

  const std::vector<Expression> input_exprs = create_all_ode_symbolic_inputs(D, J);
  const vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > > visitors;
  visitors.reserve(n_row);

  for (size_t i = 0; i < n_row; ++i) {
    std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > row;
    row.reserve(n_col);
    for (size_t j = 0; j < n_col; ++j) {
      std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > dep;
      dep.reserve(n_dep);
      for (size_t k = 0; k < n_dep; ++k) {
        dep.emplace_back(std::make_unique<LambdaRealDoubleVisitor>());
      }
      row.emplace_back(std::move(dep));
    }
    visitors.emplace_back(std::move(row));
  }

  for (size_t i = 0; i < n_row; ++i) {
    for (size_t j = 0; j < n_col; ++j) {
      for (size_t k = 0; k < n_dep; ++k) {
        auto basic = H_f[i][j][k].get_basic();
        visitors[i][j][k]->init(inputs, *basic);
      }
    }
  }

  return visitors;
}

// 4-D Tensor input
std::vector<std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > > >
build_jacobian_visitors(
  const std::vector<std::vector<std::vector<std::vector<Expression> > > > &T_f,
  const size_t D,
  const size_t J
) {
  const size_t n_row = T_f.size();
  const size_t n_col = n_row > 0 ? T_f[0].size() : 0;
  const size_t n_dep1 = n_col > 0 ? T_f[0][0].size() : 0;
  const size_t n_dep2 = n_dep1 > 0 ? T_f[0][0][0].size() : 0;

  const std::vector<Expression> input_exprs = create_all_ode_symbolic_inputs(D, J);
  const vec_basic inputs = expressions_to_vec_basic(input_exprs);

  std::vector<std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > > > visitors;
  visitors.reserve(n_row);

  for (size_t i = 0; i < n_row; ++i) {
    std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > > row;
    row.reserve(n_col);
    for (size_t j = 0; j < n_col; ++j) {
      std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > plane;
      plane.reserve(n_dep1);
      for (size_t k = 0; k < n_dep1; ++k) {
        std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > line;
        line.reserve(n_dep2);
        for (size_t l = 0; l < n_dep2; ++l) {
          line.emplace_back(std::make_unique<LambdaRealDoubleVisitor>());
        }
        plane.emplace_back(std::move(line));
      }
      row.emplace_back(std::move(plane));
    }
    visitors.emplace_back(std::move(row));
  }

  for (size_t i = 0; i < n_row; ++i) {
    for (size_t j = 0; j < n_col; ++j) {
      for (size_t k = 0; k < n_dep1; ++k) {
        for (size_t l = 0; l < n_dep2; ++l) {
          auto basic = T_f[i][j][k][l].get_basic();
          visitors[i][j][k][l]->init(inputs, *basic);
        }
      }
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
  const size_t rows = matrix.size();
  const size_t cols = rows > 0 ? matrix[0].size() : 0;
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

// For 3D tensor input
std::vector<std::vector<std::vector<std::vector<Expression> > > >
build_symbolic_jacobian(const std::vector<std::vector<std::vector<Expression> > > &T,
                        const std::vector<Expression> &inputs) {
  const size_t rows = T.size();
  const size_t cols = rows > 0 ? T[0].size() : 0;
  const size_t ndepth = cols > 0 ? T[0][0].size() : 0;

  // 4D jacobian: [rows][cols][ndepth][inputs.size()]
  std::vector<std::vector<std::vector<std::vector<Expression> > > > jacobian(
    rows, std::vector<std::vector<std::vector<Expression> > >(
      cols, std::vector<std::vector<Expression> >(
        ndepth, std::vector<Expression>(inputs.size())
      )
    )
  );

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      for (size_t k = 0; k < ndepth; ++k) {
        for (size_t l = 0; l < inputs.size(); ++l) {
          jacobian[i][j][k][l] = T[i][j][k].diff(inputs[l]);
        }
      }
    }
  }
  return jacobian;
}
