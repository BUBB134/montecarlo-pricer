#include "path_simulator.hpp"
#include <cmath>

namespace montecarlo
{
    PathSimulator::PathSimulator(RNG &rng)
        : rng_(rng)
    {
    }

    double PathSimulator::simulate_terminal(double S0, double r, double sigma, double T)
    {
        double Z = rng_.normal();
        double drift = (r - 0.5 * sigma * sigma) * T;
        double diffusion = sigma * std::sqrt(T) * Z;
        return S0 * std::exp(drift + diffusion);
    }

    std::vector<double> PathSimulator::simulate_path(double S0, double r, double sigma, double T, std::size_t steps)
    {
        std::vector<double> path;
        path.reserve(steps + 1);
        path.push_back(S0);

        if (steps == 0)
        {
            return path;
        }

        double dt = T / static_cast<double>(steps);
        double sqrt_dt = std::sqrt(dt);
        double S = S0;

            for (std::size_t i = 0; i < steps; ++i)
        {
            double z = rng_.normal();
            S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z);
            path.push_back(S);
        }
        return path;
    }
}
