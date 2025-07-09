#pragma once
#include <xtensor/containers/xarray.hpp>

template <typename T, typename Predicate>
int find_last(const xt::xtensor<T,1>& arr, Predicate pred) {
    for (int i = arr.size() - 1; i >= 0; --i) {
        if (pred(arr.flat(i))) return i;
    }
    return -1;
}


