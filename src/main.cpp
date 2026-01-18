#include "main.hpp"
#include "rng.hpp"
#include "payoff.hpp"
#include "monte_carlo.hpp"
#include "bs_analytical.hpp"
#include "timer.hpp"

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
            std::cout << "  Call control: β=" << call_result_cv.control_beta 
                      << ", E[Control]=" << call_result_cv.control_payoff_analytical
                      << ", Var.Red.=" << call_result_cv.variance_reduction_factor << "x\n";
            std::cout << "  Note: β=1.0 means SANITY CHECK (using same payoff as control, not true variance reduction)\n";
        }
        if (put_result_cv.control_variate_used) {
            std::cout << "  Put control: β=" << put_result_cv.control_beta 
                      << ", E[Control]=" << put_result_cv.control_payoff_analytical
                      << ", Var.Red.=" << put_result_cv.variance_reduction_factor << "x\n";
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

        // Format large numbers with K/M/B suffixes
        auto format_metric = [](double value) -> std::string {
            if (value >= 1e9) {
                return std::to_string(static_cast<int>(value / 1e9)) + "." +
                       std::to_string(static_cast<int>((value / 1e8)) % 10) + "B";
            } else if (value >= 1e6) {
                return std::to_string(static_cast<int>(value / 1e6)) + "." +
                       std::to_string(static_cast<int>((value / 1e5)) % 10) + "M";
            } else if (value >= 1e3) {
                return std::to_string(static_cast<int>(value / 1e3)) + "." +
                       std::to_string(static_cast<int>((value / 1e2)) % 10) + "K";
            } else {
                return std::to_string(static_cast<int>(value));
            }
        };

        std::cout << "\n======== Monte Carlo Pricer Benchmark ========\n";
        std::cout << "Paths: " << format_number(n_paths) << "\n";
        std::cout << "===============================================\n\n";

        // Benchmark single thread
        Timer timer_1;
        auto result_1 = pricer.price_by_mc_parallel(*call_payoff, S0, r, sigma, T, n_paths,
                                                     0.95, true, false, nullptr, 0.0, 1);
        double time_1 = timer_1.elapsed_seconds();

        BenchmarkResult bench_1;
        bench_1.time_seconds = time_1;
        bench_1.num_paths = n_paths;
        bench_1.num_threads = 1;
        bench_1.compute_metrics();

        std::cout << "Threads: 1\n";
        std::cout << "  Time:       " << std::fixed << std::setprecision(3) << time_1 << "s\n";
        std::cout << "  Throughput: " << format_metric(bench_1.paths_per_second) << " paths/sec\n";
        std::cout << "  Latency:    " << std::fixed << std::setprecision(2) 
                  << bench_1.nanoseconds_per_path << " ns/path\n";
        std::cout << "  Price:      " << std::fixed << std::setprecision(6) << result_1.price << "\n\n";

        // Benchmark multiple threads
        Timer timer_multi;
        auto result_multi = pricer.price_by_mc_parallel(*call_payoff, S0, r, sigma, T, n_paths,
                                                         0.95, true, false, nullptr, 0.0, max_threads);
        double time_multi = timer_multi.elapsed_seconds();

        BenchmarkResult bench_multi;
        bench_multi.time_seconds = time_multi;
        bench_multi.num_paths = n_paths;
        bench_multi.num_threads = max_threads;
        bench_multi.compute_metrics();

        double speedup = time_1 / time_multi;
        double efficiency = speedup / static_cast<double>(max_threads) * 100.0;

        std::cout << "Threads: " << max_threads << "\n";
        std::cout << "  Time:       " << std::fixed << std::setprecision(3) << time_multi << "s\n";
        std::cout << "  Throughput: " << format_metric(bench_multi.paths_per_second) << " paths/sec\n";
        std::cout << "  Latency:    " << std::fixed << std::setprecision(2) 
                  << bench_multi.nanoseconds_per_path << " ns/path\n";
        std::cout << "  Price:      " << std::fixed << std::setprecision(6) << result_multi.price << "\n";
        std::cout << "  Speedup:    " << std::fixed << std::setprecision(2) << speedup << "x\n";
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%\n\n";

        // Summary comparison
        std::cout << "===============================================\n";
        std::cout << "Performance Improvement:\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
                  << (bench_multi.paths_per_second / bench_1.paths_per_second) << "x faster\n";
        std::cout << "  Latency:    " << std::fixed << std::setprecision(2) 
                  << (bench_1.nanoseconds_per_path / bench_multi.nanoseconds_per_path) << "x lower\n";
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