#ifndef MONTECARLO_PRICER_BS_ANALYTICAL_HPP
#define MONTECARLO_PRICER_BS_ANALYTICAL_HPP

#include <cmath>

namespace montecarlo {
    double normal_cdf(double x);

    double black_scholes_call_price(double S0, double K, double r, double sigma, double T);

    double black_scholes_put_price(double S0, double K, double r, double sigma, double T);
}

#endif // MONTECARLO_PRICER_BS_ANALYTICAL_HPP