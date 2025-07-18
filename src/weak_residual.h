#ifndef WEAK_RESIDUAL_H
#define WEAK_RESIDUAL_H

#include <utility>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <symengine/lambda_double.h>
#include <xtensor-blas/xlinalg.hpp>

// f(p u, t) rhs of system, function of all variables
struct f_functor {
    std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > dx;
    size_t D;

    f_functor(
        std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > dx_,
        const size_t D_
    ) : dx(std::move(dx_)), D(D_) {
    }

    xt::xtensor<double, 1> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const {
        std::vector<double> inputs = p;
        inputs.insert(inputs.end(), u.begin(), u.end());
        inputs.emplace_back(t);
        xt::xtensor<double, 1> out = xt::empty<double>({D});
        for (std::size_t i = 0; i < D; ++i) {
            out[i] = dx[i]->call(inputs);
        }
        return out;
    }
};

// ∇f(p,u,t) ∈ ℝᴺ
struct J_f_functor final {
    std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > dx;

    explicit J_f_functor(std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > dx_) : dx(
        std::move(dx_)) {
    }

    ~J_f_functor() = default;

    xt::xtensor<double, 2> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const {
        const size_t nrows = dx.size();
        const size_t ncols = dx[0].size();

        std::vector<double> inputs = p;
        inputs.insert(inputs.end(), u.begin(), u.end());
        inputs.emplace_back(t);
        xt::xtensor<double, 2> out = xt::empty<double>({nrows, ncols});
        for (std::size_t i = 0; i < ncols; ++i) {
            for (std::size_t j = 0; j < nrows; ++j) {
                out(i, j) = dx[i][j]->call(inputs);
            }
        }
        return out;
    }
};

// ∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ
struct H_f_functor final {
    std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > dx;

    explicit H_f_functor(
        std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > dx_) : dx(
        std::move(dx_)) {
    }

    ~H_f_functor() = default;

    xt::xtensor<double, 3> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const {
        const size_t nrows = dx.size();
        const size_t ncols = dx[0].size();
        const size_t ndepth = dx[0][0].size();

        std::vector<double> inputs = p;
        inputs.insert(inputs.end(), u.begin(), u.end());
        inputs.emplace_back(t);
        xt::xtensor<double, 3> out = xt::empty<double>({nrows, ncols, ndepth});
        for (std::size_t i = 0; i < ncols; ++i) {
            for (std::size_t j = 0; j < nrows; ++j) {
                for (std::size_t k = 0; k < ndepth; ++k) {
                    out(i, j, k) = dx[i][j][k]->call(inputs);
                }
            }
        }
        return out;
    }
};


// ∇∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ x ℝᵁ
struct T_f_functor final {
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > >dx;

    explicit T_f_functor(
        std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > > dx_) : dx(
        std::move(dx_)) {
    }

    ~T_f_functor() = default;

    xt::xtensor<double, 3> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const {
        const size_t nrows = dx.size();
        const size_t ncols = dx[0].size();
        const size_t ndepth = dx[0][0].size();
        const size_t ndepth2 = dx[0][0][0].size();

        std::vector<double> inputs = p;
        inputs.insert(inputs.end(), u.begin(), u.end());
        inputs.emplace_back(t);
        xt::xtensor<double, 4> out = xt::empty<double>({nrows, ncols, ndepth, ndepth2});
        for (std::size_t i = 0; i < ncols; ++i) {
            for (std::size_t j = 0; j < nrows; ++j) {
                for (std::size_t k = 0; k < ndepth; ++k) {
                    for (std::size_t l = 0; l < ndepth; ++l) {
                        out(i, j, k, l) = dx[i][j][k][l]->call(inputs);
                    }
                }
            }
        }
        return out;
    }
};


// F(p,U,t) ∈ ℝᵐ x ℝᴰ Matrix valued function filled with u(t_0),...., u(t_m) where u(t_i) ∈ ℝᴰ
struct F_functor {
    const f_functor &f;
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;

    F_functor(
        const f_functor &f_,
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_
    ) : f(f_), U(U_), tt(tt_) {
    }

    xt::xtensor<double, 2> operator()(const std::vector<double> &p) const {
        auto F_eval = xt::zeros_like(U);
        for (int i = 0; i < U.shape()[1]; ++i) {
            auto row = xt::view(F_eval, i, xt::all());
            auto u = xt::view(U, i, xt::all());
            auto t = tt[i];
            row = f(p, u, t);
        }
        return F_eval;
    }
};


// g(p) = vec[Phi F(p,U,t)] ∈ ℝ^(mp1 x D) column wise vectorization
struct g_functor {
    const xt::xtensor<double, 2> &V_prime;
    const F_functor &F;

    g_functor(const F_functor &F_,
              const xt::xtensor<double, 2> &V_prime_): V_prime(V_prime_), F(F_) {
    }

    xt::xtensor<double, 1> operator()(const std::vector<double> &p) const {
        const auto F_eval = F(p);
        return (xt::ravel<xt::layout_type::column_major>(xt::linalg::dot(F_eval, V_prime)));
    }
};

// ∇g(p) Jacobian of g w.r.t all state variables J_f is respect to at all the time points, function of p. The data are known.
struct J_g_functor {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const J_f_functor &J_f;
    const size_t D;
    const size_t mp1;
    const size_t K;
    xt::xtensor<double, 2> V_expanded;
    size_t grad_len;

    J_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const J_f_functor &J_f_
    )
        : U(U_), tt(tt_), V(V_), J_f(J_f_),
          D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {
        V_expanded = xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3); // (K, mp1, 1, 1)
        grad_len = J_f.dx[0].size();
    }

    xt::xtensor<double, 4> operator()(
        const std::vector<double> &p
    ) const {
        xt::xtensor<double, 3> J_F({mp1, D, grad_len}); // Compute J_F: (mp1, D, gradient_len)
        for (size_t i = 0; i < mp1; ++i) {
            const double &t = tt[i];
            const auto &u = xt::view(U, i, xt::all());
            xt::view(J_F, i, xt::all(), xt::all()) = J_f(p, u, t);
        }
        // V_expanded has dimension (K, mp1, 1, 1)
        auto J_F_expanded = xt::expand_dims(J_F, 0); // (1, mp1, D, len(∇))
        auto Jg = V_expanded * J_F_expanded; // (K, mp1, D, len(∇))
        auto J_g_t = xt::transpose(xt::eval(Jg), {0, 2, 3, 1}); // (K, D, len(∇), mp1)

        return J_g_t;
    }
};

// ∇∇g(p) with respect to two difference variables
struct H_g_functor {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const H_f_functor H_f;
    const size_t D;
    const size_t mp1;
    const size_t K;
    xt::xtensor<double, 2> V_expanded;
    size_t grad1_len;
    size_t grad2_len;

    H_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const H_f_functor &H_f_
    )
        : U(U_), tt(tt_), V(V_), H_f(H_f_), D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {

        V_expanded = xt::expand_dims(xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3),4); // (K, mp1, 1, 1, 1)
        grad1_len = H_f.dx[0].size();
        grad2_len = H_f.dx[0][0].size();
    }

    xt::xtensor<double, 5> operator()(
        const std::vector<double> &p
    ) const {
        // Compute H_F with dimension (mp1, D, len(∇₁), len(∇₂))
        xt::xtensor<double, 4> H_F({mp1, D, grad1_len, grad2_len});
        for (size_t i = 0; i < mp1; ++i) {
            const double &t = tt[i];
            const auto &u = xt::view(U, i, xt::all());
            xt::view(H_F, i, xt::all(), xt::all(), xt::all()) = H_f(p, u, t);
        }
        //Compute Hg                                                         // V_expanded has dimension (K, mp1, 1, 1, 1)
        const auto H_F_expanded = xt::expand_dims(H_F, 0);    // (1, mp1, D, len(∇₁), len(∇₂)
        const auto Hg = V_expanded * H_F_expanded;             //  (K, mp1, D, len(∇₁), len(∇₂))
        const auto Hgt = xt::transpose(xt::eval(Hg), {0, 2, 3, 1, 4}); // (K, D, len(∇₁), mp1, len(∇₂))
        return Hgt;
    }
};

// ∇∇∇g(p) with respect to three difference variables
struct T_g_functor {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const T_f_functor T_f;
    const size_t D;
    const size_t mp1;
    const size_t K;
    xt::xtensor<double, 2> V_expanded;
    size_t grad1_len;
    size_t grad2_len;
    size_t grad3_len;

    T_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const T_f_functor &T_f_
    )
        : U(U_), tt(tt_), V(V_), T_f(T_f_), D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {

        V_expanded = xt::expand_dims(xt::expand_dims(xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3),4), 5); // (K, mp1, 1, 1, 1 ,1)
        grad1_len = T_f.dx[0].size();
        grad2_len = T_f.dx[0][0].size();
        grad3_len = T_f.dx[0][0][0].size();
    }

    xt::xtensor<double, 6> operator()(
        const std::vector<double> &p
    ) const {
        // Compute H_F with dimension (mp1, D, len(∇₁), len(∇₂) len(∇₃))
        xt::xtensor<double, 5> H_F({mp1, D, grad1_len, grad2_len, grad3_len});
        for (size_t i = 0; i < mp1; ++i) {
            const double &t = tt[i];
            const auto &u = xt::view(U, i, xt::all());
            xt::view(H_F, i, xt::all(), xt::all(), xt::all()) = T_f(p, u, t);
        }
        //Compute Tg                                                         // V_expanded has dimension (K, mp1, 1, 1, 1)
        const auto T_F_expanded = xt::expand_dims(H_F, 0);    // (1, mp1, D, len(∇₁), len(∇₂), len(∇₃))
        const auto Tg = V_expanded * T_F_expanded;             //  (K, mp1, D, len(∇₁), len(∇₂),len(∇₃) )
        const auto Tgt = xt::transpose(xt::eval(Tg), {0, 2, 3, 1, 4, 5}); // (K, D, len(∇₁), mp1, len(∇₂), len(∇₃))
        return Tgt;
    }
};

#endif //WEAK_RESIDUAL_H
