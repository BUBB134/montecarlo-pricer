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
}