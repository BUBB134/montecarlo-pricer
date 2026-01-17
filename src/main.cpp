#include "main.hpp"
#include "rng.hpp"
#include "payoff.hpp"
#include "monte_carlo.hpp"
#include "bs_analytical.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

namespace montecarlo
{
    void run_demo(std::size_t n_paths, uint64_t seed)
    {
        if (seed == 0)
        {
            seed = static_cast<uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
        RNG rng(seed);
        MonteCarloPricer pricer(rng);

        double S0 = 100.0;
        double K = 100.0;
        double r = 0.05;
        double sigma = 0.2;
        double T = 1.0;

        auto call_payoff = make_call(K);
        auto put_payoff = make_put(K);

        auto call_result = pricer.price_by_mc(*call_payoff, S0, r, sigma, T, n_paths);
        auto put_result = pricer.price_by_mc(*put_payoff, S0, r, sigma, T, n_paths);

        double call_bs = black_scholes_call_price(S0, K, r, sigma, T);
        double put_bs = black_scholes_put_price(S0, K, r, sigma, T);

        double conf = 0.95;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Call (MC):  " << call_result.price << "  std.err " << call_result.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << call_result.ci_lower << ", " << call_result.ci_upper << "]"
                  << "    Call (BS): " << call_bs << "\n";
        std::cout << "Put  (MC):  " << put_result.price << "  std.err " << put_result.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << put_result.ci_lower << ", " << put_result.ci_upper << "]"
                  << "    Put  (BS): " << put_bs << "\n";
    }
}

int main(int argc, char **argv)
{
    std::size_t n_paths = 100'000;
    uint64_t seed = 0;

    if (argc > 1)
    {
        try
        {
            n_paths = static_cast<std::size_t>(std::stoull(argv[1]));
        }
        catch (...)
        {
        }
    }
    if (argc > 2)
    {
        try
        {
            seed = static_cast<uint64_t>(std::stoull(argv[2]));
        }
        catch (...)
        {
        }
    }

    montecarlo::run_demo(n_paths, seed);
    return 0;
}