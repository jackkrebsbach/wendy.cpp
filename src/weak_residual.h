#ifndef WEAK_RESIDUAL_H
#define WEAK_RESIDUAL_H

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <symengine/lambda_double.h>
#include <xtensor/containers/xadapt.hpp>

// f(p u, t) rhs of system, function of all variables
struct f_functor {
    std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > dx;
    size_t D;

    f_functor(
        std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > dx_,
        size_t D_
    );

    xt::xtensor<double, 1> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const;
};

// ∇f(p,u,t) ∈ ℝᴺ
struct J_f_functor final {
    std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > dx;

    explicit J_f_functor(std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > dx_);

    ~J_f_functor() = default;

    xt::xtensor<double, 2> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const;
};

// ∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ
struct H_f_functor final {
    std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > dx;

    explicit H_f_functor(
        std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > dx_);

    ~H_f_functor() = default;

    xt::xtensor<double, 3> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const;
};


// ∇∇∇f(p,u,t) ∈ ℝᵐ x ℝᴺ x ℝᴹ x ℝᵁ
struct T_f_functor final {
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > > dx;

    explicit T_f_functor(
        std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> > > > >
        dx_);

    ~T_f_functor() = default;

    xt::xtensor<double, 4> operator()(
        const std::vector<double> &p,
        const xt::xtensor<double, 1> &u,
        const double &t
    ) const;
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
    );

    xt::xtensor<double, 2> operator()(const std::vector<double> &p) const;
};


// g(p) = vec[Phi F(p,U,t)] ∈ ℝ^(mp1 x D) column wise vectorization
struct g_functor {
    const xt::xtensor<double, 2> &V;
    const F_functor &F;

    g_functor(const F_functor &F_,
              const xt::xtensor<double, 2> &V_);

    xt::xtensor<double, 1> operator()(const std::vector<double> &p) const;
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
    xt::xtensor<double, 4> V_expanded;
    size_t grad_len;

    J_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const J_f_functor &J_f_
    );

    xt::xtensor<double, 4> operator()(
        const std::vector<double> &p
    ) const;
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
    xt::xtensor<double, 5> V_expanded;
    size_t grad1_len;
    size_t grad2_len;

    H_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const H_f_functor &H_f_
    );

    xt::xtensor<double, 5> operator()(
        const std::vector<double> &p
    ) const;
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
    xt::xtensor<double, 6> V_expanded;
    size_t grad1_len;
    size_t grad2_len;
    size_t grad3_len;

    T_g_functor(
        const xt::xtensor<double, 2> &U_,
        const xt::xtensor<double, 1> &tt_,
        const xt::xtensor<double, 2> &V_,
        const T_f_functor &T_f_
    );

    xt::xtensor<double, 6> operator()(
        const std::vector<double> &p
    ) const;
};

#endif //WEAK_RESIDUAL_H
