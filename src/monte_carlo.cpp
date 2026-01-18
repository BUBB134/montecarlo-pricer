#include "monte_carlo.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <thread>
#include <vector>
#include <numeric>
#include <random>

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
                                                double confidence_level,
                                                bool use_antithetic,
                                                bool use_control_variate,
                                                const Payoff *control_payoff,
                                                double control_payoff_analytical)
    {
        PricingResult result;
        result.samples = n_paths;
        if (n_paths == 0)
            return result;

        const double discount = std::exp(-r * T);
        double sum = 0.0;
        double sum_squared = 0.0;
        double control_sum = 0.0;
        std::size_t effective_samples = n_paths;

        const double drift = (r - 0.5 * sigma * sigma) * T;
        const double diffusion_scale = sigma * std::sqrt(T);

        if (!use_antithetic)
        {
            for (std::size_t i = 0; i < n_paths; ++i)
            {
                double Z = rng_.normal();
                double ST = S0 * std::exp(drift + diffusion_scale * Z);
                double payoff_value = payoff(ST);
                double disc = discount * payoff_value;
                sum += disc;
                sum_squared += disc * disc;

                // Compute control variate if requested
                if (use_control_variate && control_payoff)
                {
                    double control_value = (*control_payoff)(ST);
                    control_sum += discount * control_value;
                }
            }
        }
        else
        {
            std::size_t pairs = n_paths / 2;
            bool has_odd = (n_paths % 2) != 0;

            for (std::size_t i = 0; i < pairs; ++i)
            {
                double Z = rng_.normal();

                double ST1 = S0 * std::exp(drift + diffusion_scale * Z);
                double payoff1 = payoff(ST1);
                double disc1 = discount * payoff1;

                double ST2 = S0 * std::exp(drift + diffusion_scale * (-Z));
                double payoff2 = payoff(ST2);
                double disc2 = discount * payoff2;

                // Average the pair - this is ONE estimate
                double pair_avg = (disc1 + disc2) / 2.0;
                sum += pair_avg;
                sum_squared += pair_avg * pair_avg;

                // Compute control variate if requested
                if (use_control_variate && control_payoff)
                {
                    double control1 = (*control_payoff)(ST1);
                    double control2 = (*control_payoff)(ST2);
                    double control_pair_avg = (discount * control1 + discount * control2) / 2.0;
                    control_sum += control_pair_avg;
                }
            }

            if (has_odd)
            {
                double Z = rng_.normal();
                double ST = S0 * std::exp(drift + diffusion_scale * Z);
                double payoff_value = payoff(ST);
                double disc = discount * payoff_value;
                sum += disc;
                sum_squared += disc * disc;

                // Compute control variate if requested
                if (use_control_variate && control_payoff)
                {
                    double control_value = (*control_payoff)(ST);
                    control_sum += discount * control_value;
                }
            }

            // Adjust effective_samples to reflect the number of independent estimates
            effective_samples = pairs + (has_odd ? 1 : 0);
        }

        double mean = sum / static_cast<double>(effective_samples);

        // Apply control variate adjustment if enabled
        if (use_control_variate && control_payoff)
        {
            double control_mean_mc = control_sum / static_cast<double>(effective_samples);
            double beta = 1.0;  // control variate coefficient
            mean = mean + beta * (control_payoff_analytical - control_mean_mc);
            result.control_payoff_mc = control_mean_mc;
            result.control_payoff_analytical = control_payoff_analytical;
            result.control_variate_used = true;
        }

        result.price = mean;
        result.confidence_level = confidence_level;

        if (effective_samples > 1)
        {
            double variance = (sum_squared - static_cast<double>(effective_samples) * mean * mean) / static_cast<double>(effective_samples - 1);
            if (variance < 0.0 && variance > -1e-14)
                variance = 0.0;
            result.std_error = std::sqrt(variance / static_cast<double>(effective_samples));
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

        result.ci_lower = result.price - z * result.std_error;
        result.ci_upper = result.price + z * result.std_error;
        return result;
    }

    MonteCarloPricer::ThreadWorkerResult
    MonteCarloPricer::thread_worker(const Payoff &payoff,
                                    double S0,
                                    double r,
                                    double sigma,
                                    double T,
                                    std::size_t n_paths,
                                    bool use_antithetic,
                                    bool use_control_variate,
                                    const Payoff *control_payoff,
                                    uint64_t thread_seed)
    {
        ThreadWorkerResult result;

        // Create independent RNG for this thread
        RNG thread_rng(thread_seed);

        const double discount = std::exp(-r * T);
        const double drift = (r - 0.5 * sigma * sigma) * T;
        const double diffusion_scale = sigma * std::sqrt(T);

        // SIMD-friendly batch size (process multiple paths at once)
        constexpr std::size_t BATCH_SIZE = 64;

        if (!use_antithetic)
        {
            std::size_t n_batches = n_paths / BATCH_SIZE;
            std::size_t remainder = n_paths % BATCH_SIZE;

            // Structure-of-Arrays (SoA) layout for SIMD auto-vectorization
            std::vector<double> Z_batch(BATCH_SIZE);
            std::vector<double> ST_batch(BATCH_SIZE);
            std::vector<double> payoff_batch(BATCH_SIZE);
            std::vector<double> disc_batch(BATCH_SIZE);

            // Process full batches
            for (std::size_t batch = 0; batch < n_batches; ++batch)
            {
                // Generate batch of random numbers
                thread_rng.normal_batch(Z_batch.data(), BATCH_SIZE);

                // Compute stock prices (vectorizable loop)
                for (std::size_t i = 0; i < BATCH_SIZE; ++i)
                {
                    ST_batch[i] = S0 * std::exp(drift + diffusion_scale * Z_batch[i]);
                }

                // Compute payoffs (vectorizable loop)
                for (std::size_t i = 0; i < BATCH_SIZE; ++i)
                {
                    payoff_batch[i] = payoff(ST_batch[i]);
                }

                // Compute discounted values (vectorizable loop)
                for (std::size_t i = 0; i < BATCH_SIZE; ++i)
                {
                    disc_batch[i] = discount * payoff_batch[i];
                }

                // Accumulate results (vectorizable reduction)
                for (std::size_t i = 0; i < BATCH_SIZE; ++i)
                {
                    result.sum += disc_batch[i];
                    result.sum_squared += disc_batch[i] * disc_batch[i];
                }

                // Control variate if requested
                if (use_control_variate && control_payoff)
                {
                    for (std::size_t i = 0; i < BATCH_SIZE; ++i)
                    {
                        double control_value = (*control_payoff)(ST_batch[i]);
                        result.control_sum += discount * control_value;
                    }
                }
            }

            // Process remainder paths
            for (std::size_t i = 0; i < remainder; ++i)
            {
                double Z = thread_rng.normal();
                double ST = S0 * std::exp(drift + diffusion_scale * Z);
                double payoff_value = payoff(ST);
                double disc = discount * payoff_value;
                result.sum += disc;
                result.sum_squared += disc * disc;

                if (use_control_variate && control_payoff)
                {
                    double control_value = (*control_payoff)(ST);
                    result.control_sum += discount * control_value;
                }
            }
            result.effective_samples = n_paths;
        }
        else
        {
            // Antithetic variates with batch processing
            std::size_t pairs = n_paths / 2;
            bool has_odd = (n_paths % 2) != 0;

            constexpr std::size_t BATCH_SIZE_PAIRS = 32;
            std::size_t n_batches = pairs / BATCH_SIZE_PAIRS;
            std::size_t remainder_pairs = pairs % BATCH_SIZE_PAIRS;

            // Structure-of-Arrays for antithetic pairs
            std::vector<double> Z_batch(BATCH_SIZE_PAIRS);
            std::vector<double> ST1_batch(BATCH_SIZE_PAIRS);
            std::vector<double> ST2_batch(BATCH_SIZE_PAIRS);
            std::vector<double> payoff1_batch(BATCH_SIZE_PAIRS);
            std::vector<double> payoff2_batch(BATCH_SIZE_PAIRS);

            // Process full batches of pairs
            for (std::size_t batch = 0; batch < n_batches; ++batch)
            {
                thread_rng.normal_batch(Z_batch.data(), BATCH_SIZE_PAIRS);

                // Compute stock prices for both antithetic paths
                for (std::size_t i = 0; i < BATCH_SIZE_PAIRS; ++i)
                {
                    ST1_batch[i] = S0 * std::exp(drift + diffusion_scale * Z_batch[i]);
                    ST2_batch[i] = S0 * std::exp(drift + diffusion_scale * (-Z_batch[i]));
                }

                // Compute payoffs
                for (std::size_t i = 0; i < BATCH_SIZE_PAIRS; ++i)
                {
                    payoff1_batch[i] = payoff(ST1_batch[i]);
                    payoff2_batch[i] = payoff(ST2_batch[i]);
                }

                // Accumulate averaged pairs
                for (std::size_t i = 0; i < BATCH_SIZE_PAIRS; ++i)
                {
                    double disc1 = discount * payoff1_batch[i];
                    double disc2 = discount * payoff2_batch[i];
                    double pair_avg = (disc1 + disc2) / 2.0;
                    result.sum += pair_avg;
                    result.sum_squared += pair_avg * pair_avg;
                }

                if (use_control_variate && control_payoff)
                {
                    for (std::size_t i = 0; i < BATCH_SIZE_PAIRS; ++i)
                    {
                        double control1 = (*control_payoff)(ST1_batch[i]);
                        double control2 = (*control_payoff)(ST2_batch[i]);
                        double control_pair_avg = (discount * control1 + discount * control2) / 2.0;
                        result.control_sum += control_pair_avg;
                    }
                }
            }

            // Process remainder pairs
            for (std::size_t i = 0; i < remainder_pairs; ++i)
            {
                double Z = thread_rng.normal();

                double ST1 = S0 * std::exp(drift + diffusion_scale * Z);
                double payoff1 = payoff(ST1);
                double disc1 = discount * payoff1;

                double ST2 = S0 * std::exp(drift + diffusion_scale * (-Z));
                double payoff2 = payoff(ST2);
                double disc2 = discount * payoff2;

                double pair_avg = (disc1 + disc2) / 2.0;
                result.sum += pair_avg;
                result.sum_squared += pair_avg * pair_avg;

                if (use_control_variate && control_payoff)
                {
                    double control1 = (*control_payoff)(ST1);
                    double control2 = (*control_payoff)(ST2);
                    double control_pair_avg = (discount * control1 + discount * control2) / 2.0;
                    result.control_sum += control_pair_avg;
                }
            }

            // Handle odd path
            if (has_odd)
            {
                double Z = thread_rng.normal();
                double ST = S0 * std::exp(drift + diffusion_scale * Z);
                double payoff_value = payoff(ST);
                double disc = discount * payoff_value;
                result.sum += disc;
                result.sum_squared += disc * disc;

                if (use_control_variate && control_payoff)
                {
                    double control_value = (*control_payoff)(ST);
                    result.control_sum += discount * control_value;
                }
            }

            result.effective_samples = pairs + (has_odd ? 1 : 0);
        }

        return result;
    }

    PricingResult MonteCarloPricer::price_by_mc_parallel(const Payoff &payoff,
                                                        double S0,
                                                        double r,
                                                        double sigma,
                                                        double T,
                                                        std::size_t n_paths,
                                                        double confidence_level,
                                                        bool use_antithetic,
                                                        bool use_control_variate,
                                                        const Payoff *control_payoff,
                                                        double control_payoff_analytical,
                                                        std::size_t n_threads)
    {
        PricingResult result;
        result.samples = n_paths;
        if (n_paths == 0)
            return result;

        // Determine number of threads
        if (n_threads == 0)
            n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0)
            n_threads = 4;  // fallback

        // Distribute paths among threads
        std::size_t paths_per_thread = n_paths / n_threads;
        std::size_t remainder = n_paths % n_threads;

        // Create thread-safe RNG with different seeds for each thread
        std::random_device rd;
        std::vector<uint64_t> thread_seeds(n_threads);
        for (std::size_t i = 0; i < n_threads; ++i)
        {
            thread_seeds[i] = rd() + i * 12345;
        }

        // Launch threads and collect results
        std::vector<std::thread> threads;
        std::vector<ThreadWorkerResult> thread_results(n_threads);

        for (std::size_t i = 0; i < n_threads; ++i)
        {
            std::size_t this_n_paths = paths_per_thread + (i < remainder ? 1 : 0);

            threads.emplace_back([this, &payoff, S0, r, sigma, T, this_n_paths,
                                  use_antithetic, use_control_variate, control_payoff,
                                  &thread_results, i, &thread_seeds]()
            {
                thread_results[i] = thread_worker(payoff, S0, r, sigma, T,
                                                   this_n_paths, use_antithetic,
                                                   use_control_variate, control_payoff,
                                                   thread_seeds[i]);
            });
        }

        // Wait for all threads to complete
        for (auto &t : threads)
            t.join();

        // Reduce partial sums from all threads
        double sum = 0.0;
        double sum_squared = 0.0;
        double control_sum = 0.0;
        std::size_t effective_samples = 0;

        for (const auto &thread_result : thread_results)
        {
            sum += thread_result.sum;
            sum_squared += thread_result.sum_squared;
            control_sum += thread_result.control_sum;
            effective_samples += thread_result.effective_samples;
        }

        double mean = sum / static_cast<double>(effective_samples);

        // Apply control variate adjustment if enabled
        if (use_control_variate && control_payoff)
        {
            double control_mean_mc = control_sum / static_cast<double>(effective_samples);
            double beta = 1.0;  // control variate coefficient
            mean = mean + beta * (control_payoff_analytical - control_mean_mc);
            result.control_payoff_mc = control_mean_mc;
            result.control_payoff_analytical = control_payoff_analytical;
            result.control_variate_used = true;
        }

        result.price = mean;
        result.confidence_level = confidence_level;

        if (effective_samples > 1)
        {
            double variance = (sum_squared - static_cast<double>(effective_samples) * mean * mean) / static_cast<double>(effective_samples - 1);
            if (variance < 0.0 && variance > -1e-14)
                variance = 0.0;
            result.std_error = std::sqrt(variance / static_cast<double>(effective_samples));
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

        result.ci_lower = result.price - z * result.std_error;
        result.ci_upper = result.price + z * result.std_error;
        return result;
    }
}