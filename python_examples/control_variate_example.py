"""
Example: Using Control Variate with Different Strike
"""
import montecarlo_pricer as mcp

# Create pricer
pricer = mcp.MonteCarloPricer(seed=42)
config = mcp.PricingConfig()

# Setup option to price
config.S0 = 100.0
config.K = 100.0
config.r = 0.05
config.sigma = 0.2
config.T = 1.0
config.n_paths = 1_000_000
config.option_type = 'call'
config.use_antithetic = False

# Price without control variate
config.use_control_variate = False
result_baseline = pricer.price_mc_parallel(config)

# Price WITH control variate using K=95 call as control
config.use_control_variate = True
config.control_strike = 95.0  # Use K=95 call as control for K=100 call
result_cv = pricer.price_mc_parallel(config)

print(f"Target: K=100 Call")
print(f"Control: K=95 Call")
print()
print(f"Without CV: Price = {result_baseline.price:.4f}, Std Error = {result_baseline.std_error:.6f}")
print(f"With CV:    Price = {result_cv.price:.4f}, Std Error = {result_cv.std_error:.6f}")
print()
print(f"Optimal Beta: {result_cv.control_beta:.4f}")
print(f"Variance Reduction: {(result_baseline.std_error/result_cv.std_error):.2f}x")
print()
print("How it works:")
print(f"  - Monte Carlo estimates both K=100 call AND K=95 call")
print(f"  - K=95 call analytical price: {result_cv.control_payoff_analytical:.4f}")
print(f"  - K=95 call MC estimate:      {result_cv.control_payoff_mc:.4f}")
print(f"  - Adjustment = β × (analytical - MC) = {result_cv.control_beta:.4f} × {(result_cv.control_payoff_analytical - result_cv.control_payoff_mc):.4f}")
print(f"  - Final price = {result_baseline.price:.4f} + {result_cv.control_beta * (result_cv.control_payoff_analytical - result_cv.control_payoff_mc):.4f} = {result_cv.price:.4f}")

# Usage modes:
print("\n" + "="*60)
print("USAGE MODES:")
print("="*60)

print("\n1. Same strike, same type (no benefit):")
print("   config.control_strike = 100.0  # Same as K")
print("   config.control_option_type = 'auto'  # Uses 'call' (same as option_type)")

print("\n2. Different strike, same type (RECOMMENDED):")
print("   config.control_strike = 95.0   # Different strike")
print("   config.control_option_type = 'auto'  # Uses 'call' (same type)")

print("\n3. Different strike, different type:")
print("   config.control_strike = 95.0")
print("   config.control_option_type = 'put'  # Explicit control type")

print("\n4. Same strike, different type (minimal benefit):")
print("   config.control_strike = 0.0  # 0.0 means use K")
print("   config.control_option_type = 'put'")
