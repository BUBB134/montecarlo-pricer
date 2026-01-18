"""
Performance benchmarking example comparing different configurations
"""

import time

import montecarlo_pricer as mcp


def format_number(num):
    """Format large numbers with K/M/B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(int(num))

def benchmark_threads(n_paths=10_000_000):
    """Benchmark different thread counts"""
    pricer = mcp.MonteCarloPricer(seed=42)
    
    config = mcp.PricingConfig()
    config.S0 = 100.0
    config.K = 100.0
    config.r = 0.05
    config.sigma = 0.2
    config.T = 1.0
    config.n_paths = n_paths
    config.option_type = "call"
    config.use_antithetic = True
    
    print("=" * 70)
    print(f"Threading Benchmark - {format_number(n_paths)} paths")
    print("=" * 70)
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16]
    results = []
    
    for n_threads in thread_counts:
        config.n_threads = n_threads
        
        start = time.perf_counter()
        result = pricer.price_mc_parallel(config)
        elapsed = time.perf_counter() - start
        
        paths_per_sec = n_paths / elapsed
        ns_per_path = (elapsed * 1e9) / n_paths
        
        results.append({
            'threads': n_threads,
            'time': elapsed,
            'paths_per_sec': paths_per_sec,
            'ns_per_path': ns_per_path,
            'price': result.price
        })
        
        print(f"Threads: {n_threads:2d}  "
              f"Time: {elapsed:.3f}s  "
              f"Throughput: {format_number(paths_per_sec)} paths/sec  "
              f"Latency: {ns_per_path:.2f} ns/path")
    
    # Calculate speedup
    base_time = results[0]['time']
    print("\n" + "=" * 70)
    print("Speedup Analysis:")
    print("=" * 70)
    for r in results:
        speedup = base_time / r['time']
        efficiency = (speedup / r['threads']) * 100
        print(f"Threads: {r['threads']:2d}  "
              f"Speedup: {speedup:.2f}x  "
              f"Efficiency: {efficiency:.1f}%")
    print()

def benchmark_variance_reduction(n_paths=1_000_000):
    """Compare variance reduction techniques"""
    pricer = mcp.MonteCarloPricer(seed=42)
    
    config = mcp.PricingConfig()
    config.S0 = 100.0
    config.K = 100.0
    config.r = 0.05
    config.sigma = 0.2
    config.T = 1.0
    config.n_paths = n_paths
    config.option_type = "call"
    config.n_threads = 8
    
    analytical = pricer.analytical_price(config)
    
    print("=" * 70)
    print(f"Variance Reduction Comparison - {format_number(n_paths)} paths")
    print("=" * 70)
    print(f"Black-Scholes Price: {analytical:.6f}\n")
    
    # Test configurations
    configs_to_test = [
        ("Standard MC", False, False),
        ("Antithetic Variates", True, False),
        ("Control Variate", False, True),
        ("Antithetic + Control", True, True),
    ]
    
    for name, use_anti, use_cv in configs_to_test:
        config.use_antithetic = use_anti
        config.use_control_variate = use_cv
        
        start = time.perf_counter()
        result = pricer.price_mc_parallel(config)
        elapsed = time.perf_counter() - start
        
        error = abs(result.price - analytical)
        variance_reduction = (result.std_error / results[0].std_error) if name != "Standard MC" else 1.0
        
        print(f"{name:25s}  "
              f"Price: {result.price:.6f}  "
              f"Error: {error:.6f}  "
              f"Std Err: {result.std_error:.6f}  "
              f"Time: {elapsed:.3f}s")
        
        if name == "Standard MC":
            base_std_error = result.std_error
            results = [result]
        else:
            vr_factor = base_std_error / result.std_error
            print(f"{'':25s}  Variance Reduction: {vr_factor:.2f}x")
    print()

def benchmark_path_counts():
    """Compare accuracy vs speed for different path counts"""
    pricer = mcp.MonteCarloPricer(seed=42)
    
    config = mcp.PricingConfig()
    config.S0 = 100.0
    config.K = 100.0
    config.r = 0.05
    config.sigma = 0.2
    config.T = 1.0
    config.option_type = "call"
    config.use_antithetic = True
    config.n_threads = 8
    
    analytical = pricer.analytical_price(config)
    
    print("=" * 70)
    print("Path Count Analysis")
    print("=" * 70)
    print(f"Black-Scholes Price: {analytical:.6f}\n")
    
    path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    
    for n_paths in path_counts:
        config.n_paths = n_paths
        
        start = time.perf_counter()
        result = pricer.price_mc_parallel(config)
        elapsed = time.perf_counter() - start
        
        error = abs(result.price - analytical)
        error_pct = (error / analytical) * 100
        
        print(f"Paths: {format_number(n_paths):>8s}  "
              f"Price: {result.price:.6f}  "
              f"Error: {error:.6f} ({error_pct:.3f}%)  "
              f"Std Err: {result.std_error:.6f}  "
              f"Time: {elapsed:.3f}s")
    print()

def main():
    print("\n" + "=" * 70)
    print("Monte Carlo Pricer - Comprehensive Benchmarks")
    print("=" * 70 + "\n")
    
    # Run all benchmarks
    benchmark_threads(n_paths=10_000_000)
    benchmark_variance_reduction(n_paths=1_000_000)
    benchmark_path_counts()

if __name__ == "__main__":
    main()
