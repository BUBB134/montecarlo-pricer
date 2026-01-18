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
#include <thread>

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

    void run_benchmark(std::size_t n_paths, std::size_t max_threads)
    {
        if (max_threads == 0)
            max_threads = std::thread::hardware_concurrency();
        if (max_threads == 0)
            max_threads = 8;

        uint64_t seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        RNG rng(seed);
        MonteCarloPricer pricer(rng);

        double S0 = 100.0;
        double K = 100.0;
        double r = 0.05;
        double sigma = 0.2;
        double T = 1.0;

        auto call_payoff = make_call(K);

        // Format number with thousands separator
        auto format_number = [](std::size_t num) -> std::string {
            std::string result = std::to_string(num);
            int pos = result.length() - 3;
            while (pos > 0) {
                result.insert(pos, ",");
                pos -= 3;
            }
            return result;
        };

        std::cout << "\n======== Threading Benchmark ========\n";
        std::cout << "Paths: " << format_number(n_paths) << "\n\n";

        // Benchmark single thread
        auto start = std::chrono::high_resolution_clock::now();
        auto result_1 = pricer.price_by_mc_parallel(*call_payoff, S0, r, sigma, T, n_paths,
                                                     0.95, true, false, nullptr, 0.0, 1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_1 = std::chrono::duration<double>(end - start).count();

        std::cout << "Threads: 1  -> " << std::fixed << std::setprecision(2) << duration_1 << "s\n";

        // Benchmark multiple threads
        start = std::chrono::high_resolution_clock::now();
        auto result_multi = pricer.price_by_mc_parallel(*call_payoff, S0, r, sigma, T, n_paths,
                                                         0.95, true, false, nullptr, 0.0, max_threads);
        end = std::chrono::high_resolution_clock::now();
        auto duration_multi = std::chrono::duration<double>(end - start).count();

        double speedup = duration_1 / duration_multi;
        std::cout << "Threads: " << max_threads << " -> " << std::fixed << std::setprecision(2) 
                  << duration_multi << "s (" << std::fixed << std::setprecision(1) 
                  << speedup << "x speedup)\n";

        // Verify results are consistent
        std::cout << "\nPrice (1 thread):  " << std::fixed << std::setprecision(6) << result_1.price << "\n";
        std::cout << "Price (" << max_threads << " threads): " << std::fixed << std::setprecision(6) << result_multi.price << "\n";
    }
}

int main(int argc, char **argv)
{
    std::size_t n_paths = 100'000;
    uint64_t seed = 0;
    bool run_bench = false;

    if (argc > 1)
    {
        try
        {
            std::string arg = argv[1];
            if (arg == "--benchmark" || arg == "-b")
            {
                run_bench = true;
                // Check if second arg is path count
                if (argc > 2)
                {
                    n_paths = static_cast<std::size_t>(std::stoull(argv[2]));
                }
            }
            else
            {
                n_paths = static_cast<std::size_t>(std::stoull(argv[1]));
            }
        }
        catch (...)
        {
        }
    }
    if (argc > 2 && !run_bench)
    {
        try
        {
            seed = static_cast<uint64_t>(std::stoull(argv[2]));
        }
        catch (...)
        {
        }
    }

    if (run_bench)
    {
        montecarlo::run_benchmark(n_paths);
    }
    else
    {
        montecarlo::run_demo(n_paths, seed);
    }
    return 0;
}