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

        // Standard MC with antithetic sampling
        auto call_result = pricer.price_by_mc(*call_payoff, S0, r, sigma, T, n_paths, 0.95, true);
        auto put_result = pricer.price_by_mc(*put_payoff, S0, r, sigma, T, n_paths, 0.95, true);

        // MC with antithetic + control variate (using same payoff as control)
        double call_bs = black_scholes_call_price(S0, K, r, sigma, T);
        double put_bs = black_scholes_put_price(S0, K, r, sigma, T);
        
        auto call_result_cv = pricer.price_by_mc(*call_payoff, S0, r, sigma, T, n_paths, 0.95, 
                                                  true,                  // use_antithetic
                                                  true,                  // use_control_variate
                                                  call_payoff.get(),     // control_payoff
                                                  call_bs);              // control_analytical_price
        
        auto put_result_cv = pricer.price_by_mc(*put_payoff, S0, r, sigma, T, n_paths, 0.95,
                                                 true,                   // use_antithetic
                                                 true,                   // use_control_variate
                                                 put_payoff.get(),      // control_payoff
                                                 put_bs);               // control_analytical_price

        // MC without antithetic sampling
        auto call_result_no_anti = pricer.price_by_mc(*call_payoff, S0, r, sigma, T, n_paths, 0.95, false);
        auto put_result_no_anti = pricer.price_by_mc(*put_payoff, S0, r, sigma, T, n_paths, 0.95, false);

        double conf = 0.95;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "======== Antithetic Variates (no control variate) ========\n";
        std::cout << "Call (MC):  " << call_result.price << "  std.err " << call_result.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << call_result.ci_lower << ", " << call_result.ci_upper << "]"
                  << "    Call (BS): " << call_bs << "\n";
        std::cout << "Put  (MC):  " << put_result.price << "  std.err " << put_result.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << put_result.ci_lower << ", " << put_result.ci_upper << "]"
                  << "    Put  (BS): " << put_bs << "\n";

        std::cout << "\n======== Antithetic + Control Variate ========\n";
        std::cout << "Call (MC+CV): " << call_result_cv.price << "  std.err " << call_result_cv.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << call_result_cv.ci_lower << ", " << call_result_cv.ci_upper << "]"
                  << "    Call (BS): " << call_bs << "\n";
        std::cout << "Put  (MC+CV): " << put_result_cv.price << "  std.err " << put_result_cv.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << put_result_cv.ci_lower << ", " << put_result_cv.ci_upper << "]"
                  << "    Put  (BS): " << put_bs << "\n";
        if (call_result_cv.control_variate_used) {
            std::cout << "  Call control adjustment: " << (call_bs - call_result_cv.control_payoff_mc) << "\n";
        }
        if (put_result_cv.control_variate_used) {
            std::cout << "  Put control adjustment: " << (put_bs - put_result_cv.control_payoff_mc) << "\n";
        }

        std::cout << "\n======== No Antithetic Variates ========\n";
        std::cout << "Call (MC):  " << call_result_no_anti.price << "  std.err " << call_result_no_anti.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << call_result_no_anti.ci_lower << ", " << call_result_no_anti.ci_upper << "]"
                  << "    Call (BS): " << call_bs << "\n";
        std::cout << "Put  (MC):  " << put_result_no_anti.price << "  std.err " << put_result_no_anti.std_error
                  << "  CI(" << (conf * 100.0) << "%)=[" << put_result_no_anti.ci_lower << ", " << put_result_no_anti.ci_upper << "]"
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