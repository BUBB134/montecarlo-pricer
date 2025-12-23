#ifndef MONTOCARLO_PRICER_MONTE_CARLO_HPP
#define MONTOCARLO_PRICER_MONTE_CARLO_HPP

#include "rng.hpp"
#include "payoff.hpp"
#include <cstddef>

namespace montecarlo
{
    struct PricingResult
    {
        double price{0.0};
        double std_error{0.0};
        std::size_t samples{0};
    };

    class MonteCarloPricer
    {
    public:
        explicit MonteCarloPricer(RNG &rng);

        PricingResult price_by_mc(const Payoff &payoff,
                                  double S0,
                                  double r,
                                  double sigma,
                                  double T,
                                  std::size_t n_paths);

    private:
        RNG &rng_;
    };
}

#endif // MONTECARLO_PRICER_MONTE_CARLO_HPP