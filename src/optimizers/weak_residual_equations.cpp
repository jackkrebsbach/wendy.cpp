#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <symengine/lambda_double.h>

//The rhs of the system: dx is the symengine built function to call with
// inputs are in order [p_0, p_1, ..., p_j, u_0, u_1, ..., u_d, t]
// that is a vector with the parameters, state of the system, and time
xt::xarray<double> f(
    std::vector<double>& p, // parameters of the system
    const xt::xtensor<double, 1>& u, // state for one time point
    double &t,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& dx
    ){

    std::vector<double> inputs = p;
    inputs.insert(p.end(), u.begin(), u.end());
    inputs.emplace_back(t);

    xt::xtensor<double, 1> out = xt::empty<double>({dx.size()});
    for (std::size_t i =0; i < dx.size(); ++i) {
        out[i] = dx[i].call(inputs);
    }
    return out;
}

// g(p) = vec[Phi F(p,U,t)]
template <typename F>
xt::xtensor<double, 1> g(
    std::vector<double>& p,
    xt::xtensor<double, 2>& U,
    xt::xtensor<double, 1>& tt,
    xt::xtensor<double, 2>& V,
    F& f,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& dx
    ) {

    const double n_points = U.shape()[0];
    const double D = U.shape()[1];

    xt::xtensor<double, 1> F_eval = xt::empty<double>({n_points, D});

    for (std::size_t i = 0; i < U.shape()[0]; ++i) {
        auto t = tt[i];
        auto u_t = xt::view(U, i, xt::all());
        auto f_t = xt::view(F_eval, i, xt::all());
        f_t = f(p, u_t, t, dx);
    }

    auto V_F_eval = xt::linalg::dot(V, F_eval);

    return xt::ravel<xt::layout_type::column_major>(V_F_eval);
}

// r(p) = g(p) - b
// b = vec[\dot{Phi}U]

template <typename F>
xt::xtensor<double,1> r(
    std::vector<double>& p,
    xt::xtensor<double, 2>& U,
    xt::xtensor<double, 1>& tt,
    xt::xtensor<double, 2>& V,
    xt::xtensor<double, 2>& V_prime,
    F& f,
    std::vector<SymEngine::LambdaRealDoubleVisitor>& dx
    ) {
    auto V_prime_U = xt::linalg::dot(V_prime, U);
    auto b = xt::ravel<xt::layout_type::column_major>(V_prime_U);
    auto g = g(p, U, tt, V, f,dx);
    auto r = g-b;
    return r;
}

// ∇g_u(p) gradient of g w.r.t the vectorized data
xt::xtensor<double,2> grad_g_u() {

}