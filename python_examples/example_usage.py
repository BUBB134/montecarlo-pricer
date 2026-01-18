"""
Example usage of the Monte Carlo Pricer Python bindings
"""

import montecarlo_pricer as mcp


def main():
    # Create a pricer instance
    pricer = mcp.MonteCarloPricer(seed=42)
    
    # Create a configuration object
    config = mcp.PricingConfig()
    config.S0 = 100.0           # Initial stock price
    config.K = 100.0            # Strike price
    config.r = 0.05             # Risk-free rate (5%)
    config.sigma = 0.2          # Volatility (20%)
    config.T = 1.0              # Time to maturity (1 year)
    config.n_paths = 1_000_000  # Number of paths
    config.option_type = "call" # Call option
    config.use_antithetic = True
    config.n_threads = 0        # Auto-detect threads
    
    print("=" * 60)
    print("Monte Carlo Option Pricer - Python Interface")
    print("=" * 60)
    print(f"Configuration: {config}")
    print()
    
    # Get analytical price for comparison
    analytical = pricer.analytical_price(config)
    print(f"Black-Scholes Analytical Price: {analytical:.6f}")
    print()
    
    # Price using single-threaded Monte Carlo
    print("Single-threaded Monte Carlo:")
    result_single = pricer.price_mc(config)
    print(f"  Price:     {result_single.price:.6f}")
    print(f"  Std Error: {result_single.std_error:.6f}")
    print(f"  95% CI:    [{result_single.ci_lower:.6f}, {result_single.ci_upper:.6f}]")
    print(f"  Samples:   {result_single.samples:,}")
    print()
    
    # Price using multi-threaded Monte Carlo
    print("Multi-threaded Monte Carlo:")
    result_parallel = pricer.price_mc_parallel(config)
    print(f"  Price:     {result_parallel.price:.6f}")
    print(f"  Std Error: {result_parallel.std_error:.6f}")
    print(f"  95% CI:    [{result_parallel.ci_lower:.6f}, {result_parallel.ci_upper:.6f}]")
    print(f"  Samples:   {result_parallel.samples:,}")
    print()
    
    # Compute Greeks
    print("Computing Greeks (using parallel pricing)...")
    greeks = pricer.compute_greeks(config, use_parallel=True)
    print(f"  Delta: {greeks.delta:.6f}  (sensitivity to stock price)")
    print(f"  Gamma: {greeks.gamma:.6f}  (rate of change of Delta)")
    print(f"  Vega:  {greeks.vega:.6f}  (sensitivity to volatility)")
    print(f"  Theta: {greeks.theta:.6f}  (time decay)")
    print(f"  Rho:   {greeks.rho:.6f}  (sensitivity to interest rate)")
    print()
    
    # Put option example
    print("=" * 60)
    print("Put Option Example")
    print("=" * 60)
    config.option_type = "put"
    config.n_paths = 500_000
    
    result_put = pricer.price_mc_parallel(config)
    analytical_put = pricer.analytical_price(config)
    
    print(f"Analytical Price: {analytical_put:.6f}")
    print(f"MC Price:         {result_put.price:.6f}")
    print(f"Std Error:        {result_put.std_error:.6f}")
    print(f"95% CI:           [{result_put.ci_lower:.6f}, {result_put.ci_upper:.6f}]")
    print()
    
    # Control variate example
    print("=" * 60)
    print("With Control Variate (Call Option)")
    print("=" * 60)
    config.option_type = "call"
    config.use_control_variate = True
    config.n_paths = 100_000
    
    result_cv = pricer.price_mc_parallel(config)
    print(f"MC Price (with CV): {result_cv.price:.6f}")
    print(f"Std Error:          {result_cv.std_error:.6f}")
    print(f"Control Variate Used: {result_cv.control_variate_used}")
    if result_cv.control_variate_used:
        print(f"Control MC:         {result_cv.control_payoff_mc:.6f}")
        print(f"Control Analytical: {result_cv.control_payoff_analytical:.6f}")
    print()

if __name__ == "__main__":
    main()
