#include <wendy/weak_residual.h>

#include <utility>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <symengine/lambda_double.h>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>

// f(p u, t) rhs of system, function of all variables
f_functor::f_functor(
    std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > dx_,
    const size_t D_
) : dx(std::move(dx_)), D(D_) {
}

xt::xtensor<double, 1> f_functor::operator()(
    const std::vector<double> &p,
    const xt::xtensor<double, 1> &u,
    const double &t
) const {
    std::vector<double> inputs;
    inputs.reserve(p.size() + u.size() + 1);
    inputs.insert(inputs.end(), p.begin(), p.end());
    inputs.insert(inputs.end(), u.begin(), u.end());
    inputs.emplace_back(t);

    xt::xtensor<double, 1> out = xt::empty<double>({D});

    for (std::size_t i = 0; i < D; ++i) {
        out(i) = dx[i]->call(inputs);
    }
    return out;
}

// ∇f(p,u,t) ∈ ℝᴺ
J_f_functor::J_f_functor(
    std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > dx_) : dx(std::move(dx_)),
    n_rows(dx.size()), n_cols(dx[0].size()) {
}

xt::xtensor<double, 2> J_f_functor::operator()(
    const std::vector<double> &p,
    const xt::xtensor<double, 1> &u,
    const double &t
) const {
    std::vector<double> inputs;
    inputs.reserve(p.size() + u.size() + 1);
    inputs.insert(inputs.end(), p.begin(), p.end());
    inputs.insert(inputs.end(), u.begin(), u.end());
    inputs.emplace_back(t);

    xt::xtensor<double, 2> out = xt::empty<double>({n_rows, n_cols});

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            out(i, j) = dx[i][j]->call(inputs);
        }
    }
    return out;
}

// ∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ
H_f_functor::H_f_functor(
    std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > >
    dx_) : dx(std::move(dx_)), n_rows(dx.size()), n_cols(dx[0].size()), n_depth(dx[0][0].size()) {
}

xt::xtensor<double, 3> H_f_functor::operator()(
    const std::vector<double> &p,
    const xt::xtensor<double, 1> &u,
    const double &t
) const {
    std::vector<double> inputs;
    inputs.reserve(p.size() + u.size() + 1);
    inputs.insert(inputs.end(), p.begin(), p.end());
    inputs.insert(inputs.end(), u.begin(), u.end());
    inputs.emplace_back(t);

    xt::xtensor<double, 3> out = xt::empty<double>({n_rows, n_cols, n_depth});

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            for (std::size_t k = 0; k < n_depth; ++k) {
                out(i, j, k) = dx[i][j][k]->call(inputs);
            }
        }
    }
    return out;
}

// ∇∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ x ℝᵁ
T_f_functor::T_f_functor(
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > > dx_)
    : dx(std::move(dx_)), n_rows(dx.size()), n_cols(dx[0].size()), n_depth(dx[0][0].size()),
      n_depth2(dx[0][0][0].size()) {
}

xt::xtensor<double, 4> T_f_functor::operator()(
    const std::vector<double> &p,
    const xt::xtensor<double, 1> &u,
    const double &t
) const {
    std::vector<double> inputs;
    inputs.reserve(p.size() + u.size() + 1);
    inputs.insert(inputs.end(), p.begin(), p.end());
    inputs.insert(inputs.end(), u.begin(), u.end());
    inputs.emplace_back(t);

    xt::xtensor<double, 4> out = xt::empty<double>({n_rows, n_cols, n_depth, n_depth2});

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            for (std::size_t k = 0; k < n_depth; ++k) {
                for (std::size_t l = 0; l < n_depth2; ++l) {
                    out(i, j, k, l) = dx[i][j][k][l]->call(inputs);
                }
            }
        }
    }
    return out;
}

// F(p,U,t) ∈ ℝᵐ x ℝᴰ Matrix valued function filled with u(t_0),...., u(t_m) where u(t_i) ∈ ℝᴰ
F_functor::F_functor(
    const f_functor &f_,
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_
) : f(f_), U(U_), tt(tt_) {
}

xt::xtensor<double, 2> F_functor::operator()(const std::vector<double> &p) const {
    auto F_eval = xt::zeros_like(U);

    for (int i = 0; i < U.shape()[0]; ++i) {
        auto u = xt::row(U, i);
        auto t = tt[i];
        xt::row(F_eval, i) = f(p, u, t);
    }
    return xt::eval(F_eval);
}

// g(p) = vec[Phi F(p,U,t)] ∈ ℝ^(mp1 x D) column wise vectorization
g_functor::g_functor(const F_functor &F_, const xt::xtensor<double, 2> &V_): V(V_), F(F_) {
}

xt::xtensor<double, 1> g_functor::operator()(const std::vector<double> &p) const {
    return xt::eval((xt::ravel(xt::linalg::dot(V, F(p)))));
}