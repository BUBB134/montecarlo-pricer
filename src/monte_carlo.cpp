#include "monte_carlo.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

namespace montecarlo
{
    MonteCarloPricer::MonteCarloPricer(RNG &rng)
        : rng_(rng)
    {
    }

    PricingResult MonteCarloPricer::price_by_mc(const Payoff &payoff,
                                                double S0,
                                                double r,
                                                double sigma,
                                                double T,
                                                std::size_t n_paths,
                                                double confidence_level)
    {
        PricingResult result;
        result.samples = n_paths;
        if (n_paths == 0)
            return result;

        const double discount = std::exp(-r * T);
        double sum = 0.0;
        double sum_squared = 0.0;

        const double drift = (r - 0.5 * sigma * sigma) * T;
        const double diffusion_scale = sigma * std::sqrt(T);

        for (std::size_t i = 0; i < n_paths; ++i)
        {
            double Z = rng_.normal();
            double ST = S0 * std::exp(drift + diffusion_scale * Z);
            double payoff_value = payoff(ST);
            double disc = discount * payoff_value;
            sum += disc;
            sum_squared += disc * disc;
        }

        double mean = sum / static_cast<double>(n_paths);
        result.price = mean;
        result.confidence_level = confidence_level;

        if (n_paths > 1)
        {
            double variance = (sum_squared - static_cast<double>(n_paths) * mean * mean) / static_cast<double>(n_paths - 1);
            if (variance < 0.0 && variance > -1e-14)
                variance = 0.0;
            result.std_error = std::sqrt(variance / static_cast<double>(n_paths));
        }
        else
        {
            result.std_error = std::numeric_limits<double>::infinity();
        }

        double z = 1.96;
        if (std::abs(confidence_level - 0.90) < 1e-12)
            z = 1.645;
        else if (std::abs(confidence_level - 0.95) < 1e-12)
            z = 1.96;
        else if (std::abs(confidence_level - 0.99) < 1e-12)
            z = 2.576;
        if (!(confidence_level > 0.0 && confidence_level < 1.0))
        {
            z = 1.96;
            result.confidence_level = 0.95;
        }

        result.ci_lower = mean - z * result.std_error;
        result.ci_upper = mean + z * result.std_error;
        return result;
    }
}
