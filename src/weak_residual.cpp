#include "weak_residual.h"

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
) : dx(std::move(dx_)), D(D_) {}

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
        out.unchecked(i) = dx[i]->call(inputs);
    }
    return out;
}

// ∇f(p,u,t) ∈ ℝᴺ
J_f_functor::J_f_functor(std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > dx_) :
dx(std::move(dx_)), n_rows(dx.size()), n_cols(dx[0].size()) {}

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
    for (std::size_t j = 0; j < n_cols; ++j) {
       for (std::size_t i = 0; i < n_rows; ++i) {
            out.unchecked(i, j) = dx[i][j]->call(inputs);
        }
    }
    return out;
}

// ∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ
H_f_functor::H_f_functor(
    std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > dx_) :
    dx(std::move(dx_)), n_rows(dx.size()), n_cols(dx[0].size()), n_depth(dx[0][0].size()) {}

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
                out.unchecked(i, j, k) = dx[i][j][k]->call(inputs);
            }
        }
    }
    return out;
}

// ∇∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ x ℝᵁ
T_f_functor::T_f_functor(
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > > dx_)
    : dx(std::move(dx_)), n_rows(dx.size()), n_cols(dx[0].size()), n_depth(dx[0][0].size()), n_depth2(dx[0][0][0].size()) {
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
                    out.unchecked(i, j, k, l) = dx[i][j][k][l]->call(inputs);
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
    return F_eval;
}

// g(p) = vec[Phi F(p,U,t)] ∈ ℝ^(mp1 x D) column wise vectorization
g_functor::g_functor(const F_functor &F_, const xt::xtensor<double, 2> &V_): V(V_), F(F_) {}

xt::xtensor<double, 1> g_functor::operator()(const std::vector<double> &p) const {
    return (xt::ravel(xt::linalg::dot(V, F(p))));
}

// ∇g(p) Jacobian of g w.r.t all state variables J_f is respect to at all the time points, function of p. The data are known.
J_g_functor::J_g_functor(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const J_f_functor &J_f_
)
    : U(U_), tt(tt_), V(V_), J_f(J_f_),
      D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {
    V_expanded = xt::expand_dims(xt::expand_dims(V, 2), 3); // (K, mp1, 1, 1)
    grad_len = J_f.dx[0].size();
    // V_expanded = xt::broadcast(xt::reshape_view(V, std::vector<size_t>({K, mp1, 1, 1})), {K, mp1, D, grad_len});
}

xt::xtensor<double, 4> J_g_functor::operator()(
    const std::vector<double> &p
) const {
    xt::xtensor<double, 3> J_F({mp1, D, grad_len}); // Compute J_F: (mp1, D, gradient_len)

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::row(U,i);
        xt::view(J_F, i, xt::all(), xt::all()) = J_f(p, u, t);
    }                                    // V_expanded has dimension (K, mp1, 1, 1)
    auto J_F_expanded = xt::expand_dims(J_F, 0); // (1, mp1, D, len(∇))
    // auto J_F_expanded = xt::broadcast(J_F, {K, mp1, D, grad_len});
    auto Jg = xt::eval(V_expanded * J_F_expanded);          // (K, mp1, D, len(∇))
    auto J_g_t = xt::transpose(Jg, {0, 2, 1, 3}); // (K, D, mp1, len(∇))

    return J_g_t;
}

// ∇∇g(p) with respect to two difference variables
H_g_functor::H_g_functor(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const H_f_functor &H_f_
)
    : U(U_), tt(tt_), V(V_), H_f(H_f_), D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {
    V_expanded = xt::expand_dims(xt::expand_dims(xt::expand_dims(V, 2), 3), 4); // (K, mp1, 1, 1, 1)
    grad1_len = H_f.dx[0].size();
    grad2_len = H_f.dx[0][0].size();
}

xt::xtensor<double, 5> H_g_functor::operator()(
    const std::vector<double> &p
) const {
    // Compute H_F with dimension (mp1, D, len(∇₁), len(∇₂))
    xt::xtensor<double, 4> H_F({mp1, D, grad1_len, grad2_len});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::row(U, i);
        xt::view(H_F, i, xt::all(), xt::all(), xt::all()) = H_f(p, u, t);
    }
    //Compute Hg                               // V_expanded has dimension (K, mp1, 1,       1, 1)
    const auto H_F_expanded = xt::expand_dims(H_F, 0); // (1, mp1, D, len(∇₁), len(∇₂)
    const auto Hg = V_expanded * H_F_expanded; //  (K, mp1, D, len(∇₁), len(∇₂))
    auto Hgt = xt::transpose(xt::eval(Hg), {0, 2, 1, 3, 4}); // (K, D, mp1, len(∇₁), len(∇₂))
    return Hgt;
}

// ∇∇∇g(p) with respect to three difference variables
T_g_functor::T_g_functor(
    const xt::xtensor<double, 2> &U_,
    const xt::xtensor<double, 1> &tt_,
    const xt::xtensor<double, 2> &V_,
    const T_f_functor &T_f_
)
    : U(U_), tt(tt_), V(V_), T_f(T_f_), D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {
    V_expanded = xt::expand_dims(xt::expand_dims(xt::expand_dims(xt::expand_dims(V, 2), 3), 4), 5);
    // (K, mp1, 1, 1, 1 ,1)
    grad1_len = T_f.dx[0].size();
    grad2_len = T_f.dx[0][0].size();
    grad3_len = T_f.dx[0][0][0].size();
}


xt::xtensor<double, 6> T_g_functor::operator()(
    const std::vector<double> &p
) const {
    // Compute H_F with dimension (mp1, D, len(∇₁), len(∇₂) len(∇₃))
    xt::xtensor<double, 5> H_F({mp1, D, grad1_len, grad2_len, grad3_len});

    for (size_t i = 0; i < mp1; ++i) {
        const double &t = tt[i];
        const auto &u = xt::row(U, i);
        xt::view(H_F, i, xt::all(), xt::all(), xt::all()) = T_f(p, u, t);
    }
    //Compute Tg                                                        // V_expanded has dimension (K, mp1, 1, 1, 1)
    const auto T_F_expanded = xt::expand_dims(H_F, 0); // (1, mp1, D, len(∇₁), len(∇₂), len(∇₃))
    const auto Tg = V_expanded * T_F_expanded; //  (K, mp1, D, len(∇₁), len(∇₂),len(∇₃) )
    auto Tgt = xt::transpose(xt::eval(Tg), {0, 2, 1, 3, 4, 5}); // (K, D, mp1, len(∇₁), len(∇₂), len(∇₃))
    return Tgt;
}

