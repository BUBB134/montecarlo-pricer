#include "bs_analytical.hpp"
#include <algorithm> // for std::max
namespace montecarlo
{
    inline double normal_cdf(double x)
    {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        return 0.5 * (1.0 + std::erf(x * inv_sqrt2));
    }

    double black_scholes_call_price(double S0, double K, double r, double sigma, double T)
    {
        if (T <= 0.0)
        {
            return std::max(S0 - K, 0.0);
        }

        if (sigma <= 0.0)
        {
            double ST = S0 * std::exp(r * T);
            return std::exp(-r * T) * std::max(ST - K, 0.0);
        }
        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;

        return S0 * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    }

    double black_scholes_put_price(double S0, double K, double r, double sigma, double T)
    {
        if (T <= 0.0)
        {
            return std::max(K - S0, 0.0);
        }

        if (sigma <= 0.0)
        {
            double ST = S0 * std::exp(r * T);
            return std::exp(-r * T) * std::max(K - ST, 0.0);
        }

        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;

        return K * std::exp(-r * T) * normal_cdf(-d2) - S0 * normal_cdf(-d1);
    }
}