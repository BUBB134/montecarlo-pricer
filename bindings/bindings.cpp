#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "rng.hpp"
#include "payoff.hpp"
#include "monte_carlo.hpp"
#include "bs_analytical.hpp"

#include <memory>
#include <string>

namespace py = pybind11;
using namespace montecarlo;

// Greeks computation helper
struct Greeks
{
    double delta{0.0};
    double gamma{0.0};
    double vega{0.0};
    double theta{0.0};
    double rho{0.0};
};

// Configuration object for pricing
struct PricingConfig
{
    double S0{100.0};           // Initial stock price
    double K{100.0};            // Strike price
    double r{0.05};             // Risk-free rate
    double sigma{0.2};          // Volatility
    double T{1.0};              // Time to maturity
    std::size_t n_paths{100000}; // Number of paths
    double confidence_level{0.95}; // Confidence level
    bool use_antithetic{true};  // Use antithetic variates
    bool use_control_variate{false}; // Use control variate
    std::size_t n_threads{0};   // Number of threads (0 = auto-detect)
    std::string option_type{"call"}; // "call" or "put"
};

// Wrapper class for Python interface
class MonteCarloPricerPy
{
public:
    MonteCarloPricerPy(uint64_t seed = 0)
    {
        if (seed == 0)
            seed = std::random_device{}();
        rng_ = std::make_unique<RNG>(seed);
        pricer_ = std::make_unique<MonteCarloPricer>(*rng_);
    }

    // Simple price_mc interface
    PricingResult price_mc(const PricingConfig& config)
    {
        auto payoff = create_payoff(config);
        
        if (config.use_control_variate)
        {
            auto control_payoff = create_payoff(config);
            double control_analytical = compute_analytical_price(config);
            
            return pricer_->price_by_mc(
                *payoff, config.S0, config.r, config.sigma, config.T,
                config.n_paths, config.confidence_level,
                config.use_antithetic, config.use_control_variate,
                control_payoff.get(), control_analytical
            );
        }
        else
        {
            return pricer_->price_by_mc(
                *payoff, config.S0, config.r, config.sigma, config.T,
                config.n_paths, config.confidence_level,
                config.use_antithetic, false, nullptr, 0.0
            );
        }
    }

    // Parallel price_mc interface
    PricingResult price_mc_parallel(const PricingConfig& config)
    {
        auto payoff = create_payoff(config);
        
        if (config.use_control_variate)
        {
            auto control_payoff = create_payoff(config);
            double control_analytical = compute_analytical_price(config);
            
            return pricer_->price_by_mc_parallel(
                *payoff, config.S0, config.r, config.sigma, config.T,
                config.n_paths, config.confidence_level,
                config.use_antithetic, config.use_control_variate,
                control_payoff.get(), control_analytical, config.n_threads
            );
        }
        else
        {
            return pricer_->price_by_mc_parallel(
                *payoff, config.S0, config.r, config.sigma, config.T,
                config.n_paths, config.confidence_level,
                config.use_antithetic, false, nullptr, 0.0, config.n_threads
            );
        }
    }

    // Compute Greeks using finite differences
    Greeks compute_greeks(const PricingConfig& config, bool use_parallel = true)
    {
        Greeks greeks;
        
        // Base price
        double base_price = use_parallel ? 
            price_mc_parallel(config).price : 
            price_mc(config).price;
        
        // Delta: dV/dS (first derivative w.r.t. spot)
        const double dS = config.S0 * 0.01; // 1% bump
        PricingConfig config_up = config;
        config_up.S0 = config.S0 + dS;
        PricingConfig config_down = config;
        config_down.S0 = config.S0 - dS;
        
        double price_up = use_parallel ? price_mc_parallel(config_up).price : price_mc(config_up).price;
        double price_down = use_parallel ? price_mc_parallel(config_down).price : price_mc(config_down).price;
        
        greeks.delta = (price_up - price_down) / (2.0 * dS);
        
        // Gamma: d²V/dS² (second derivative w.r.t. spot)
        greeks.gamma = (price_up - 2.0 * base_price + price_down) / (dS * dS);
        
        // Vega: dV/dσ (derivative w.r.t. volatility)
        const double dsigma = 0.01; // 1% bump
        PricingConfig config_vega = config;
        config_vega.sigma = config.sigma + dsigma;
        double price_vega = use_parallel ? price_mc_parallel(config_vega).price : price_mc(config_vega).price;
        greeks.vega = (price_vega - base_price) / dsigma;
        
        // Theta: dV/dT (derivative w.r.t. time)
        const double dT = -1.0 / 365.0; // -1 day
        PricingConfig config_theta = config;
        config_theta.T = config.T + dT;
        if (config_theta.T > 0)
        {
            double price_theta = use_parallel ? price_mc_parallel(config_theta).price : price_mc(config_theta).price;
            greeks.theta = (price_theta - base_price) / dT;
        }
        
        // Rho: dV/dr (derivative w.r.t. interest rate)
        const double dr = 0.01; // 1% bump
        PricingConfig config_rho = config;
        config_rho.r = config.r + dr;
        double price_rho = use_parallel ? price_mc_parallel(config_rho).price : price_mc(config_rho).price;
        greeks.rho = (price_rho - base_price) / dr;
        
        return greeks;
    }

    // Get analytical price for comparison
    double analytical_price(const PricingConfig& config)
    {
        return compute_analytical_price(config);
    }

private:
    std::unique_ptr<RNG> rng_;
    std::unique_ptr<MonteCarloPricer> pricer_;

    std::unique_ptr<Payoff> create_payoff(const PricingConfig& config)
    {
        if (config.option_type == "call")
            return make_call(config.K);
        else if (config.option_type == "put")
            return make_put(config.K);
        else
            throw std::runtime_error("Unknown option type: " + config.option_type);
    }

    double compute_analytical_price(const PricingConfig& config)
    {
        if (config.option_type == "call")
            return black_scholes_call_price(config.S0, config.K, config.r, config.sigma, config.T);
        else if (config.option_type == "put")
            return black_scholes_put_price(config.S0, config.K, config.r, config.sigma, config.T);
        else
            throw std::runtime_error("Unknown option type: " + config.option_type);
    }
};

PYBIND11_MODULE(montecarlo_pricer, m)
{
    m.doc() = "Monte Carlo Option Pricer with Multi-threading and SIMD optimization";

    // PricingConfig class
    py::class_<PricingConfig>(m, "PricingConfig")
        .def(py::init<>())
        .def_readwrite("S0", &PricingConfig::S0, "Initial stock price")
        .def_readwrite("K", &PricingConfig::K, "Strike price")
        .def_readwrite("r", &PricingConfig::r, "Risk-free rate")
        .def_readwrite("sigma", &PricingConfig::sigma, "Volatility")
        .def_readwrite("T", &PricingConfig::T, "Time to maturity (years)")
        .def_readwrite("n_paths", &PricingConfig::n_paths, "Number of Monte Carlo paths")
        .def_readwrite("confidence_level", &PricingConfig::confidence_level, "Confidence level for CI")
        .def_readwrite("use_antithetic", &PricingConfig::use_antithetic, "Use antithetic variates")
        .def_readwrite("use_control_variate", &PricingConfig::use_control_variate, "Use control variate")
        .def_readwrite("n_threads", &PricingConfig::n_threads, "Number of threads (0=auto)")
        .def_readwrite("option_type", &PricingConfig::option_type, "Option type: 'call' or 'put'")
        .def("__repr__", [](const PricingConfig &c) {
            return "PricingConfig(S0=" + std::to_string(c.S0) + 
                   ", K=" + std::to_string(c.K) +
                   ", r=" + std::to_string(c.r) +
                   ", sigma=" + std::to_string(c.sigma) +
                   ", T=" + std::to_string(c.T) +
                   ", n_paths=" + std::to_string(c.n_paths) +
                   ", option_type='" + c.option_type + "')";
        });

    // PricingResult class
    py::class_<PricingResult>(m, "PricingResult")
        .def(py::init<>())
        .def_readwrite("price", &PricingResult::price, "Option price")
        .def_readwrite("std_error", &PricingResult::std_error, "Standard error")
        .def_readwrite("samples", &PricingResult::samples, "Number of samples")
        .def_readwrite("ci_lower", &PricingResult::ci_lower, "Confidence interval lower bound")
        .def_readwrite("ci_upper", &PricingResult::ci_upper, "Confidence interval upper bound")
        .def_readwrite("confidence_level", &PricingResult::confidence_level, "Confidence level")
        .def_readwrite("control_payoff_mc", &PricingResult::control_payoff_mc, "Control variate MC value")
        .def_readwrite("control_payoff_analytical", &PricingResult::control_payoff_analytical, "Control variate analytical value")
        .def_readwrite("control_variate_used", &PricingResult::control_variate_used, "Was control variate used")
        .def("__repr__", [](const PricingResult &r) {
            return "PricingResult(price=" + std::to_string(r.price) + 
                   ", std_error=" + std::to_string(r.std_error) +
                   ", CI=[" + std::to_string(r.ci_lower) + ", " + std::to_string(r.ci_upper) + "])";
        });

    // Greeks class
    py::class_<Greeks>(m, "Greeks")
        .def(py::init<>())
        .def_readwrite("delta", &Greeks::delta, "Delta: dV/dS")
        .def_readwrite("gamma", &Greeks::gamma, "Gamma: d²V/dS²")
        .def_readwrite("vega", &Greeks::vega, "Vega: dV/dσ")
        .def_readwrite("theta", &Greeks::theta, "Theta: dV/dT")
        .def_readwrite("rho", &Greeks::rho, "Rho: dV/dr")
        .def("__repr__", [](const Greeks &g) {
            return "Greeks(delta=" + std::to_string(g.delta) + 
                   ", gamma=" + std::to_string(g.gamma) +
                   ", vega=" + std::to_string(g.vega) +
                   ", theta=" + std::to_string(g.theta) +
                   ", rho=" + std::to_string(g.rho) + ")";
        });

    // MonteCarloPricerPy class
    py::class_<MonteCarloPricerPy>(m, "MonteCarloPricer")
        .def(py::init<uint64_t>(), py::arg("seed") = 0,
             "Initialize Monte Carlo pricer with optional seed")
        .def("price_mc", &MonteCarloPricerPy::price_mc, py::arg("config"),
             "Price option using single-threaded Monte Carlo")
        .def("price_mc_parallel", &MonteCarloPricerPy::price_mc_parallel, py::arg("config"),
             "Price option using multi-threaded Monte Carlo")
        .def("compute_greeks", &MonteCarloPricerPy::compute_greeks, 
             py::arg("config"), py::arg("use_parallel") = true,
             "Compute Greeks using finite differences")
        .def("analytical_price", &MonteCarloPricerPy::analytical_price, py::arg("config"),
             "Get Black-Scholes analytical price for comparison");

    // Standalone functions
    m.def("black_scholes_call", &black_scholes_call_price,
          py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"),
          "Black-Scholes analytical call price");
    
    m.def("black_scholes_put", &black_scholes_put_price,
          py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"),
          "Black-Scholes analytical put price");
}
