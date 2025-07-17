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


// Matrix valued function filled with u(t_0),...., u(t_m) where u(t_i) ∈ ℝᴰ
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


// g(p) = vec[Phi F(p,U,t)] column wise vectorization
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

    J_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const J_f_functor &J_f_
    )
        : U(U_), tt(tt_), V(V_), J_f(J_f_),
          D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0]) {
        V_expanded = xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3); // (K, mp1, 1, 1)
    }

    xt::xtensor<double, 4> operator()(
        const std::vector<double> &p
    ) const {
        xt::xtensor<double, 3> J_F({mp1, D, D}); // Compute J_F: (mp1, D, D)
        for (size_t i = 0; i < mp1; ++i) {
            const double &t = tt[i];
            const auto &u = xt::view(U, i, xt::all());
            auto JFi = xt::view(J_F, i, xt::all(), xt::all());
            JFi = J_f(p, u, t);
        }
        //V_expanded has dimension (K, mp1, 1, 1)
        auto J_F_expanded = xt::expand_dims(J_F, 0); // (1, mp1, D, D)
        auto Jg = V_expanded * J_F_expanded; // (K, mp1, D, D)
        auto J_g_t = xt::transpose(xt::eval(Jg), {0, 2, 1, 3}); // (K, D, mp1, D)

        return J_g_t;
    }
};

// ∇∇g(p) gradient with respect to two difference variables
struct H_g_functor {
    const xt::xtensor<double, 2> &U;
    const xt::xtensor<double, 1> &tt;
    const xt::xtensor<double, 2> &V;
    const J_f_functor &Ju_f;
    const size_t D;
    const size_t mp1;
    const size_t K;
    xt::xtensor<double, 2> V_expanded;

    std::optional<std::vector<size_t>> transpose_axes;
    std::optional<std::vector<size_t>> reshape_shape;

    H_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        J_f_functor &Ju_f_
    )
        : U(U_), tt(tt_), V(V_), Ju_f(Ju_f_),
          D(U_.shape()[1]), mp1(U_.shape()[0]), K(V_.shape()[0])
   {
        V_expanded = xt::expand_dims(xt::expand_dims(xt::transpose(V), 2), 3); // (K, mp1, 1, 1)
    }

    xt::xtensor<double, 2> operator()(
        const std::vector<double> &p
    ) const {
        // Compute Ju_F: (mp1, D, D)
        xt::xtensor<double, 3> Ju_F({mp1, D, D});
        for (size_t i = 0; i < mp1; ++i) {
            const double &t = tt[i];
            const auto &u = xt::view(U, i, xt::all());
            auto JuFi = xt::view(Ju_F, i, xt::all(), xt::all());
            JuFi = Ju_f(p, u, t);
        }
        //V_expanded has dimension (K, mp1, 1, 1)
        auto Ju_F_expanded = xt::expand_dims(Ju_F, 0); // (1, mp1, D, D)
        auto Jug = V_expanded * Ju_F_expanded; // (K, mp1, D, D)
        auto Ju_g_t = xt::transpose(xt::eval(Jug), {0, 2, 1, 3}); // (K, D, mp1, D)
        xt::xtensor<double, 2> Ju_g = xt::reshape_view(Ju_g_t, {K * D, D * mp1}); // (K*D, D*mp1)

        return Ju_g;

    }
};

#endif //WEAK_RESIDUAL_H
