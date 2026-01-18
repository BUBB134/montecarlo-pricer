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

        // Confidence interval
        double ci_lower{0.0};
        double ci_upper{0.0};
        double confidence_level{0.95};

        // Control variate diagnostics
        double control_payoff_mc{0.0};
        double control_payoff_analytical{0.0};
        bool control_variate_used{false};
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
                                  std::size_t n_paths,
                                  double confidence_level = 0.95,
                                  bool use_antithetic = true,
                                  bool use_control_variate = false,
                                  const Payoff *control_payoff = nullptr,
                                  double control_payoff_analytical = 0.0);

    private:
        RNG &rng_;
    };
}

#endif // MONTECARLO_PRICER_MONTE_CARLO_HPP