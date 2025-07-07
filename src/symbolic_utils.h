#ifndef SYMBOLIC_UTILS_H
#define SYMBOLIC_UTILS_H

#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

std::vector<SymEngine::Expression>
create_symbolic_system(const std::vector<std::string> &f);

std::vector<Expression> create_symbolic_vars(const std::string &base_name,
                                             size_t count);

std::vector<SymEngine::Expression> create_all_symbolic_inputs(size_t J, size_t D);

std::vector<std::vector<Expression> >
compute_jacobian(const std::vector<Expression> &system,
                 const std::vector<Expression> &inputs);

vec_basic expressions_to_vec_basic(const std::vector<Expression> &exprs);

std::vector<LambdaRealDoubleVisitor>
build_symbolic_system(const std::vector<Expression> &dx, size_t D, size_t J);

std::vector<std::vector<std::vector<Expression> > >
compute_jacobian(const std::vector<std::vector<Expression> > &matrix,
                 const std::vector<Expression> &inputs);
#endif
