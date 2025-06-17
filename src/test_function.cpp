
#include <algorithm>
#include <cmath>

double MIN_CONST = 0.0001;

double phi(double t, double a, double eta = 9) {
  return (std::exp(-eta / std::max(1 - std::pow((t / a), 2), MIN_CONST)));
}
