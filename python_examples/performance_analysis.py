"""
Comprehensive performance analysis for Monte Carlo Pricer
Generates detailed reports and visualizations
"""

import sys
import time
from collections import defaultdict

import montecarlo_pricer as mcp


def format_number(num):
    """Format large numbers with K/M/B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))


class PerformanceAnalyzer:
    """Comprehensive performance analysis for Monte Carlo pricer"""
    
    def __init__(self, seed=42):
        self.pricer = mcp.MonteCarloPricer(seed=seed)
        self.results = defaultdict(list)
        
    def analyze_threading_scalability(self, n_paths=10_000_000, max_threads=None):
        """Analyze performance scaling with thread count"""
        print("\n" + "=" * 80)
        print("THREADING SCALABILITY ANALYSIS")
        print("=" * 80)
        print(f"Paths: {format_number(n_paths)}")
        print()
        
        config = mcp.PricingConfig()
        config.S0 = 100.0
        config.K = 100.0
        config.r = 0.05
        config.sigma = 0.2
        config.T = 1.0
        config.n_paths = n_paths
        config.option_type = "call"
        config.use_antithetic = True
        
        # Detect available cores
        import os
        max_cores = os.cpu_count() or 8
        if max_threads is None:
            thread_counts = [1, 2, 4, 8]
            if max_cores >= 16:
                thread_counts.extend([16, max_cores])
            elif max_cores > 8:
                thread_counts.append(max_cores)
        else:
            thread_counts = [2**i for i in range(int(max_threads).bit_length()) if 2**i <= max_threads]
            if max_threads not in thread_counts:
                thread_counts.append(max_threads)
            thread_counts = sorted([1] + thread_counts)
        
        print(f"{'Threads':<8} {'Time (s)':<10} {'Throughput':<15} {'Latency':<15} {'Speedup':<10} {'Efficiency':<12} {'Price':<12}")
        print("-" * 80)
        
        base_time = None
        for n_threads in thread_counts:
            config.n_threads = n_threads
            
            # Warm-up run
            if n_threads == 1:
                _ = self.pricer.price_mc_parallel(config)
            
            # Actual benchmark
            start = time.perf_counter()
            result = self.pricer.price_mc_parallel(config)
            elapsed = time.perf_counter() - start
            
            if base_time is None:
                base_time = elapsed
                
            throughput = n_paths / elapsed
            latency_ns = (elapsed * 1e9) / n_paths
            speedup = base_time / elapsed
            efficiency = (speedup / n_threads) * 100
            
            self.results['threading'].append({
                'threads': n_threads,
                'time': elapsed,
                'throughput': throughput,
                'latency_ns': latency_ns,
                'speedup': speedup,
                'efficiency': efficiency,
                'price': result.price
            })
            
            print(f"{n_threads:<8} {elapsed:<10.3f} {format_number(throughput) + ' p/s':<15} "
                  f"{latency_ns:<15.2f} {speedup:<10.2f} {efficiency:<12.1f} {result.price:<12.6f}")
        
        # Analysis
        print("\n" + "-" * 80)
        best = max(self.results['threading'], key=lambda x: x['throughput'])
        print(f"Best Performance: {best['threads']} threads at {format_number(best['throughput'])} paths/sec")
        print(f"Maximum Speedup: {max(r['speedup'] for r in self.results['threading']):.2f}x")
        print(f"Best Efficiency: {max(r['efficiency'] for r in self.results['threading']):.1f}%")
        
    def analyze_path_convergence(self, thread_count=8):
        """Analyze convergence and accuracy vs path count"""
        print("\n" + "=" * 80)
        print("PATH COUNT CONVERGENCE ANALYSIS")
        print("=" * 80)
        
        config = mcp.PricingConfig()
        config.S0 = 100.0
        config.K = 100.0
        config.r = 0.05
        config.sigma = 0.2
        config.T = 1.0
        config.option_type = "call"
        config.use_antithetic = True
        config.n_threads = thread_count
        
        analytical = self.pricer.analytical_price(config)
        print(f"Black-Scholes Analytical Price: {analytical:.6f}")
        print()
        
        path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
        
        print(f"{'Paths':<12} {'MC Price':<12} {'Error':<12} {'Error %':<10} {'Std Err':<12} {'Time (s)':<10} {'Throughput':<15}")
        print("-" * 80)
        
        for n_paths in path_counts:
            config.n_paths = n_paths
            
            start = time.perf_counter()
            result = self.pricer.price_mc_parallel(config)
            elapsed = time.perf_counter() - start
            
            error = abs(result.price - analytical)
            error_pct = (error / analytical) * 100
            throughput = n_paths / elapsed
            
            self.results['convergence'].append({
                'n_paths': n_paths,
                'price': result.price,
                'error': error,
                'error_pct': error_pct,
                'std_error': result.std_error,
                'time': elapsed,
                'throughput': throughput
            })
            
            print(f"{format_number(n_paths):<12} {result.price:<12.6f} {error:<12.6f} {error_pct:<10.4f} "
                  f"{result.std_error:<12.6f} {elapsed:<10.3f} {format_number(throughput) + ' p/s':<15}")
        
        print("\n" + "-" * 80)
        print(f"Convergence to analytical: {analytical:.6f}")
        print(f"Best accuracy: {min(r['error'] for r in self.results['convergence']):.6f} "
              f"at {format_number(max(self.results['convergence'], key=lambda x: x['n_paths'])['n_paths'])} paths")
        
    def analyze_variance_reduction(self, n_paths=1_000_000, thread_count=8):
        """Compare variance reduction techniques"""
        print("\n" + "=" * 80)
        print("VARIANCE REDUCTION ANALYSIS")
        print("=" * 80)
        print(f"Paths: {format_number(n_paths)}, Threads: {thread_count}")
        print()
        
        config = mcp.PricingConfig()
        config.S0 = 100.0
        config.K = 100.0
        config.r = 0.05
        config.sigma = 0.2
        config.T = 1.0
        config.n_paths = n_paths
        config.option_type = "call"
        config.n_threads = thread_count
        
        analytical = self.pricer.analytical_price(config)
        print(f"Black-Scholes Price: {analytical:.6f}")
        print()
        
        techniques = [
            ("Standard MC", False, False),
            ("Antithetic Variates", True, False),
            ("Control Variate (β=1)", False, True),
            ("Antithetic + Control", True, True),
        ]
        
        print(f"{'Technique':<28} {'Price':<12} {'Error':<12} {'Std Error':<12} {'VR Factor':<10} {'Time (s)':<10}")
        print("-" * 92)
        
        base_std_error = None
        for name, use_anti, use_cv in techniques:
            config.use_antithetic = use_anti
            config.use_control_variate = use_cv
            
            start = time.perf_counter()
            result = self.pricer.price_mc_parallel(config)
            elapsed = time.perf_counter() - start
            
            error = abs(result.price - analytical)
            
            if base_std_error is None:
                base_std_error = result.std_error
                vr_factor = 1.0
            else:
                vr_factor = base_std_error / result.std_error
            
            self.results['variance_reduction'].append({
                'technique': name,
                'price': result.price,
                'error': error,
                'std_error': result.std_error,
                'vr_factor': vr_factor,
                'time': elapsed,
                'control_beta': getattr(result, 'control_beta', None) if use_cv else None,
                'variance_reduction_factor': getattr(result, 'variance_reduction_factor', None) if use_cv else None,
                'control_expectation': getattr(result, 'control_payoff_analytical', None) if use_cv else None
            })
            
            print(f"{name:<28} {result.price:<12.6f} {error:<12.6f} {result.std_error:<12.6f} "
                  f"{vr_factor:<10.2f} {elapsed:<10.3f}")
            
            # Show control variate diagnostics
            if use_cv and result.control_variate_used:
                if hasattr(result, 'control_beta') and hasattr(result, 'variance_reduction_factor'):
                    print(f"  {'└─ Control variate:':<28} β={result.control_beta:.2f}, "
                          f"E[Control]={getattr(result, 'control_payoff_analytical', 0.0):.6f}, "
                          f"Var.Red.={result.variance_reduction_factor:.2f}x")
                    print(f"  {'   Note:':<28} β=1.0 means SANITY CHECK (not true variance reduction)")
                else:
                    print(f"  {'└─ Control variate:':<28} Enabled (rebuild module for diagnostics)")
        
        print("\n" + "-" * 92)
        best = min(self.results['variance_reduction'], key=lambda x: x['std_error'])
        print(f"Best Technique: {best['technique']} (Variance Reduction: {best['vr_factor']:.2f}x)")
        print(f"Standard Error Improvement: {base_std_error/best['std_error']:.2f}x")
        
    def analyze_greeks_accuracy(self, n_paths=1_000_000, thread_count=8):
        """Analyze Greeks computation accuracy"""
        print("\n" + "=" * 80)
        print("GREEKS ACCURACY ANALYSIS")
        print("=" * 80)
        print(f"Paths: {format_number(n_paths)}, Threads: {thread_count}")
        print()
        
        config = mcp.PricingConfig()
        config.S0 = 100.0
        config.K = 100.0
        config.r = 0.05
        config.sigma = 0.2
        config.T = 1.0
        config.n_paths = n_paths
        config.option_type = "call"
        config.n_threads = thread_count
        config.use_antithetic = True
        
        print("Computing Greeks (this may take a moment)...")
        start = time.perf_counter()
        greeks = self.pricer.compute_greeks(config, use_parallel=True)
        elapsed = time.perf_counter() - start
        
        print(f"\nGreeks Computation Time: {elapsed:.2f}s")
        print()
        print(f"{'Greek':<10} {'Value':<15} {'Description':<50}")
        print("-" * 80)
        print(f"{'Delta':<10} {greeks.delta:<15.6f} {'Sensitivity to stock price (dV/dS)':<50}")
        print(f"{'Gamma':<10} {greeks.gamma:<15.6f} {'Rate of change of Delta (d²V/dS²)':<50}")
        print(f"{'Vega':<10} {greeks.vega:<15.6f} {'Sensitivity to volatility (dV/dσ)':<50}")
        print(f"{'Theta':<10} {greeks.theta:<15.6f} {'Time decay (dV/dT)':<50}")
        print(f"{'Rho':<10} {greeks.rho:<15.6f} {'Sensitivity to interest rate (dV/dr)':<50}")
        
        self.results['greeks'] = {
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'vega': greeks.vega,
            'theta': greeks.theta,
            'rho': greeks.rho,
            'computation_time': elapsed
        }
        
    def analyze_option_types(self, n_paths=1_000_000, thread_count=8):
        """Compare call vs put option pricing"""
        print("\n" + "=" * 80)
        print("OPTION TYPE COMPARISON")
        print("=" * 80)
        
        config = mcp.PricingConfig()
        config.S0 = 100.0
        config.K = 100.0
        config.r = 0.05
        config.sigma = 0.2
        config.T = 1.0
        config.n_paths = n_paths
        config.n_threads = thread_count
        config.use_antithetic = True
        
        print(f"{'Type':<10} {'MC Price':<12} {'BS Price':<12} {'Error':<12} {'Std Error':<12} {'CI Width':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        for opt_type in ["call", "put"]:
            config.option_type = opt_type
            
            analytical = self.pricer.analytical_price(config)
            
            start = time.perf_counter()
            result = self.pricer.price_mc_parallel(config)
            elapsed = time.perf_counter() - start
            
            error = abs(result.price - analytical)
            ci_width = result.ci_upper - result.ci_lower
            
            print(f"{opt_type.capitalize():<10} {result.price:<12.6f} {analytical:<12.6f} {error:<12.6f} "
                  f"{result.std_error:<12.6f} {ci_width:<12.6f} {elapsed:<10.3f}")
        
        # Put-Call Parity Check
        print("\n" + "-" * 80)
        print("Put-Call Parity Verification:")
        config.option_type = "call"
        call_result = self.pricer.price_mc_parallel(config)
        config.option_type = "put"
        put_result = self.pricer.price_mc_parallel(config)
        
        import math
        parity_lhs = call_result.price - put_result.price
        parity_rhs = config.S0 - config.K * math.exp(-config.r * config.T)
        parity_error = abs(parity_lhs - parity_rhs)
        
        print(f"C - P = {parity_lhs:.6f}")
        print(f"S - K*exp(-rT) = {parity_rhs:.6f}")
        print(f"Parity Error: {parity_error:.6f}")
        
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 80)
        
        if 'threading' in self.results and self.results['threading']:
            print("\n>>> Threading Performance")
            best_thread = max(self.results['threading'], key=lambda x: x['throughput'])
            single_thread = next(r for r in self.results['threading'] if r['threads'] == 1)
            print(f"  Single-threaded: {format_number(single_thread['throughput'])} paths/sec")
            print(f"  Best multi-threaded: {format_number(best_thread['throughput'])} paths/sec ({best_thread['threads']} threads)")
            print(f"  Maximum speedup: {best_thread['speedup']:.2f}x")
            print(f"  Peak efficiency: {max(r['efficiency'] for r in self.results['threading']):.1f}%")
        
        if 'convergence' in self.results and self.results['convergence']:
            print("\n>>> Convergence Analysis")
            smallest = min(self.results['convergence'], key=lambda x: x['n_paths'])
            largest = max(self.results['convergence'], key=lambda x: x['n_paths'])
            print(f"  Error at {format_number(smallest['n_paths'])} paths: {smallest['error']:.6f} ({smallest['error_pct']:.3f}%)")
            print(f"  Error at {format_number(largest['n_paths'])} paths: {largest['error']:.6f} ({largest['error_pct']:.3f}%)")
            print(f"  Accuracy improvement: {smallest['error']/largest['error']:.2f}x")
        
        if 'variance_reduction' in self.results and self.results['variance_reduction']:
            print("\n>>> Variance Reduction")
            base = self.results['variance_reduction'][0]
            best = min(self.results['variance_reduction'], key=lambda x: x['std_error'])
            print(f"  Standard MC std error: {base['std_error']:.6f}")
            print(f"  Best technique: {best['technique']}")
            print(f"  Best std error: {best['std_error']:.6f}")
            print(f"  Variance reduction: {best['vr_factor']:.2f}x")
        
        if 'greeks' in self.results:
            print("\n>>> Greeks Computation")
            print(f"  Computation time: {self.results['greeks']['computation_time']:.2f}s")
            print(f"  Delta: {self.results['greeks']['delta']:.6f}")
            print(f"  Gamma: {self.results['greeks']['gamma']:.6f}")
            print(f"  Vega: {self.results['greeks']['vega']:.6f}")
        
        print("\n" + "=" * 80)


def main():
    """Run comprehensive performance analysis"""
    print("\n" + "=" * 80)
    print("MONTE CARLO PRICER - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print("\nThis analysis will benchmark:")
    print("  1. Threading scalability")
    print("  2. Path count convergence")
    print("  3. Variance reduction techniques")
    print("  4. Greeks computation")
    print("  5. Option types comparison")
    print("\nThis may take several minutes to complete...")
    
    analyzer = PerformanceAnalyzer(seed=42)
    
    try:
        # Run all analyses
        analyzer.analyze_threading_scalability(n_paths=10_000_000)
        analyzer.analyze_path_convergence(thread_count=8)
        analyzer.analyze_variance_reduction(n_paths=1_000_000, thread_count=8)
        analyzer.analyze_greeks_accuracy(n_paths=500_000, thread_count=8)
        analyzer.analyze_option_types(n_paths=1_000_000, thread_count=8)
        
        # Generate summary
        analyzer.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
