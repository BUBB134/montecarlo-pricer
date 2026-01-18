# Python Bindings for Monte Carlo Pricer

High-performance Monte Carlo option pricer with multi-threading and SIMD optimization, accessible from Python.

## Installation

### Prerequisites

```bash
pip install scikit-build-core pybind11
```

### Build and Install

```bash
# From the project root directory
pip install .

# For development (editable install)
pip install -e .
```

## Quick Start

```python
import montecarlo_pricer as mcp

# Create pricer
pricer = mcp.MonteCarloPricer(seed=42)

# Configure pricing
config = mcp.PricingConfig()
config.S0 = 100.0           # Stock price
config.K = 100.0            # Strike price
config.r = 0.05             # Risk-free rate
config.sigma = 0.2          # Volatility
config.T = 1.0              # Time to maturity
config.n_paths = 1_000_000  # Number of paths
config.option_type = "call" # "call" or "put"

# Price the option
result = pricer.price_mc_parallel(config)
print(f"Price: {result.price:.6f}")
print(f"95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")

# Compute Greeks
greeks = pricer.compute_greeks(config)
print(f"Delta: {greeks.delta:.6f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Vega: {greeks.vega:.6f}")
```

## API Reference

### Classes

#### `MonteCarloPricer`

Main pricer class for Monte Carlo simulation.

**Methods:**
- `__init__(seed=0)`: Initialize with optional random seed
- `price_mc(config)`: Single-threaded Monte Carlo pricing
- `price_mc_parallel(config)`: Multi-threaded Monte Carlo pricing
- `compute_greeks(config, use_parallel=True)`: Compute option Greeks using finite differences
- `analytical_price(config)`: Get Black-Scholes analytical price

#### `PricingConfig`

Configuration object for pricing parameters.

**Attributes:**
- `S0`: Initial stock price (default: 100.0)
- `K`: Strike price (default: 100.0)
- `r`: Risk-free rate (default: 0.05)
- `sigma`: Volatility (default: 0.2)
- `T`: Time to maturity in years (default: 1.0)
- `n_paths`: Number of Monte Carlo paths (default: 100,000)
- `confidence_level`: Confidence level for CI (default: 0.95)
- `use_antithetic`: Use antithetic variates (default: True)
- `use_control_variate`: Use control variate (default: False)
- `n_threads`: Number of threads, 0=auto (default: 0)
- `option_type`: "call" or "put" (default: "call")

#### `PricingResult`

Result object from pricing.

**Attributes:**
- `price`: Option price
- `std_error`: Standard error
- `samples`: Number of samples used
- `ci_lower`: Lower bound of confidence interval
- `ci_upper`: Upper bound of confidence interval
- `confidence_level`: Confidence level used
- `control_variate_used`: Whether control variate was used
- `control_payoff_mc`: Control variate MC value
- `control_payoff_analytical`: Control variate analytical value

#### `Greeks`

Greeks calculation result.

**Attributes:**
- `delta`: Sensitivity to stock price (∂V/∂S)
- `gamma`: Rate of change of delta (∂²V/∂S²)
- `vega`: Sensitivity to volatility (∂V/∂σ)
- `theta`: Time decay (∂V/∂T)
- `rho`: Sensitivity to interest rate (∂V/∂r)

### Standalone Functions

- `black_scholes_call(S0, K, r, sigma, T)`: Analytical call price
- `black_scholes_put(S0, K, r, sigma, T)`: Analytical put price

## Examples

### Basic Pricing

```python
import montecarlo_pricer as mcp

pricer = mcp.MonteCarloPricer()
config = mcp.PricingConfig()
config.n_paths = 1_000_000

# Call option
result = pricer.price_mc_parallel(config)
print(f"Call Price: {result.price:.6f}")

# Put option
config.option_type = "put"
result = pricer.price_mc_parallel(config)
print(f"Put Price: {result.price:.6f}")
```

### Variance Reduction

```python
# Antithetic variates (default: enabled)
config.use_antithetic = True

# Control variate (uses Black-Scholes as control)
config.use_control_variate = True

result = pricer.price_mc_parallel(config)
print(f"Variance Reduced Price: {result.price:.6f}")
print(f"Standard Error: {result.std_error:.6f}")
```

### Computing Greeks

```python
greeks = pricer.compute_greeks(config, use_parallel=True)

print(f"Delta: {greeks.delta:.6f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Vega:  {greeks.vega:.6f}")
print(f"Theta: {greeks.theta:.6f}")
print(f"Rho:   {greeks.rho:.6f}")
```

### Performance Benchmarking

```python
import time

config.n_paths = 10_000_000

# Single-threaded
config.n_threads = 1
start = time.perf_counter()
result = pricer.price_mc_parallel(config)
time_1 = time.perf_counter() - start

# Multi-threaded (auto-detect cores)
config.n_threads = 0
start = time.perf_counter()
result = pricer.price_mc_parallel(config)
time_multi = time.perf_counter() - start

speedup = time_1 / time_multi
print(f"Speedup: {speedup:.2f}x")
```

## Features

- **Multi-threading**: Automatic thread detection or manual specification
- **SIMD Optimization**: Batch processing with structure-of-arrays layout
- **Variance Reduction**: Antithetic variates and control variates
- **Greeks Computation**: Delta, Gamma, Vega, Theta, Rho
- **Confidence Intervals**: Configurable confidence levels
- **High Performance**: Optimized C++ backend with AVX2/AVX-512 support

## Performance

On a typical multi-core system (e.g., 8+ cores), expect:
- **Throughput**: 100M+ paths/second
- **Speedup**: 6-8x with 8 threads
- **Accuracy**: Converges to Black-Scholes analytical price

## Examples Directory

See `python_examples/` for complete examples:
- `example_usage.py`: Basic usage and features
- `benchmark.py`: Performance benchmarking
- `convergence_and_speed.py`: Convergence analysis

## Building from Source

```bash
# Clone repository
git clone <repository-url>
cd montecarlo-pricer

# Install build dependencies
pip install scikit-build-core pybind11

# Build and install
pip install .
```

### Build Options

```bash
# Debug build
pip install . --config-settings=cmake.build-type=Debug

# Verbose build
pip install . --verbose

# Clean rebuild
pip install . --no-build-isolation --force-reinstall
```

## Requirements

- Python 3.8+
- C++17 compiler
- CMake 3.15+
- pybind11
- scikit-build-core

## License

[Your License Here]
