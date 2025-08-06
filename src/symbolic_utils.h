#ifndef SYMBOLIC_UTILS_H
#define SYMBOLIC_UTILS_H

#include <wendy/wendy.h>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>

using namespace SymEngine;

std::vector<SymEngine::Expression>
build_symbolic_f(const std::vector<std::string> &f, size_t D, NoiseDist noise_dist);

std::vector<Expression> lognormal_transform(const std::vector<Expression> &dx);


std::vector<Expression> create_symbolic_vars(const std::string &base_name,
                                             size_t count);

std::vector<SymEngine::Expression> create_all_ode_symbolic_inputs(size_t D, size_t J);

vec_basic expressions_to_vec_basic(const std::vector<Expression> &exprs);

std::vector<std::shared_ptr<LambdaRealDoubleVisitor> >
build_f_visitors(const std::vector<Expression> &dx, size_t D, size_t J);

std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > >
build_jacobian_visitors(const std::vector<std::vector<Expression> > &J_f, size_t D, size_t J);

std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > >
build_jacobian_visitors(const std::vector<std::vector<std::vector<Expression> > > &H_f, size_t D, size_t J);

std::vector<std::vector<std::vector<std::vector<std::shared_ptr<LambdaRealDoubleVisitor> > > > >
build_jacobian_visitors(const std::vector<std::vector<std::vector<std::vector<Expression> > > > &T_f, size_t D,
                        size_t J);

std::vector<std::vector<Expression> >
build_symbolic_jacobian(const std::vector<Expression> &system,
                        const std::vector<Expression> &inputs);

std::vector<std::vector<std::vector<Expression> > >
build_symbolic_jacobian(const std::vector<std::vector<Expression> > &matrix,
                        const std::vector<Expression> &inputs);

std::vector<std::vector<std::vector<std::vector<Expression> > > >
build_symbolic_jacobian(const std::vector<std::vector<std::vector<Expression> > > &T,
                        const std::vector<Expression> &inputs);

#endif
