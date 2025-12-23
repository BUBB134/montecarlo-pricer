#ifndef MONTECARLO_PATH_SIMULATOR_HPP
#define MONTECARLO_PATH_SIMULATOR_HPP

#include "rng.hpp"

namespace montecarlo
{
    class PathSimulator
    {
    public:
        explicit PathSimulator(RNG &rng);

        double simulate_terminal(double S0, double r, double sigma, double T);

        std::vector<double> simulate_path(double S0, double r, double sigma, double T, std::size_t steps);

    private:
        RNG &rng_;
    };
}
#endif // MONTECARLO_PATH_SIMULATOR_HPP