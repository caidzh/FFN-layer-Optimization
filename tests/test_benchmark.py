"""
Benchmark tests for GEGLU FFN implementations.

This module compares the performance of baseline PyTorch implementation
and CUDA optimized implementation, measuring speedup across different batch sizes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import numpy as np
from tabulate import tabulate
from models.ffn_baseline import GEGLU_FFN


# Test configuration
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
BATCH_SIZES = [4, 8, 16, 32, 64, 128]
WARMUP_ITERS = 10
BENCHMARK_ITERS = 100


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def benchmark_baseline(batch_sizes, warmup_iters=10, benchmark_iters=100):
    """
    Benchmark the baseline PyTorch implementation.
    
    Args:
        batch_sizes (list): List of batch sizes to test
        warmup_iters (int): Number of warmup iterations
        benchmark_iters (int): Number of benchmark iterations
        
    Returns:
        dict: Dictionary mapping batch size to average time in milliseconds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
    ffn.eval()
    
    results = {}
    
    with torch.no_grad():
        for B in batch_sizes:
            x = torch.randn(B, HIDDEN_SIZE, device=device)
            
            # Warmup
            for _ in range(warmup_iters):
                _ = ffn(x)
            
            # Benchmark
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            times = []
            for _ in range(benchmark_iters):
                start = time.perf_counter()
                y = ffn(x)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[B] = {
                'avg': avg_time,
                'std': std_time,
                'min': min_time,
                'max': max_time
            }
    
    return results


def benchmark_cuda(batch_sizes, warmup_iters=10, benchmark_iters=100):
    """
    Benchmark the CUDA kernel implementation.
    
    Args:
        batch_sizes (list): List of batch sizes to test
        warmup_iters (int): Number of warmup iterations
        benchmark_iters (int): Number of benchmark iterations
        
    Returns:
        dict: Dictionary mapping batch size to average time in milliseconds,
              or None if CUDA kernel is not available
    """
    try:
        from cuda_kernel import ffn_cuda_forward
    except ImportError:
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return None
    
    set_seed(42)
    ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
    ffn.eval()
    
    # Get weights
    weights = ffn.get_weights()
    Wu = weights['Wu'].t().contiguous()  # [4096, 12288]
    Wv = weights['Wv'].t().contiguous()
    Wo = weights['Wo'].t().contiguous()  # [12288, 4096]
    
    results = {}
    
    with torch.no_grad():
        for B in batch_sizes:
            x = torch.randn(B, HIDDEN_SIZE, device=device)
            
            # Warmup
            for _ in range(warmup_iters):
                _ = ffn_cuda_forward(x, Wu, Wv, Wo)
            
            # Benchmark
            torch.cuda.synchronize()
            
            times = []
            for _ in range(benchmark_iters):
                start = time.perf_counter()
                y = ffn_cuda_forward(x, Wu, Wv, Wo)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[B] = {
                'avg': avg_time,
                'std': std_time,
                'min': min_time,
                'max': max_time
            }
    
    return results


def print_results(baseline_results, cuda_results=None):
    """
    Print benchmark results in a formatted table.
    
    Args:
        baseline_results (dict): Baseline benchmark results
        cuda_results (dict): CUDA benchmark results (optional)
    """
    print("\n" + "="*90)
    print(" BENCHMARK RESULTS")
    print("="*90)
    print(f"\nConfiguration:")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Intermediate size: {INTERMEDIATE_SIZE}")
    print(f"  Warmup iterations: {WARMUP_ITERS}")
    print(f"  Benchmark iterations: {BENCHMARK_ITERS}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Prepare table data
    headers = ["Batch Size", "Baseline (ms)", "Baseline Std"]
    table_data = []
    
    if cuda_results is not None:
        headers.extend(["CUDA (ms)", "CUDA Std", "Speedup", "Status"])
        
        for B in BATCH_SIZES:
            baseline = baseline_results[B]
            cuda = cuda_results[B]
            speedup = baseline['avg'] / cuda['avg']
            
            # Check if speedup target is met (3x)
            status = "✓" if speedup >= 3.0 else "✗"
            
            table_data.append([
                B,
                f"{baseline['avg']:.4f}",
                f"{baseline['std']:.4f}",
                f"{cuda['avg']:.4f}",
                f"{cuda['std']:.4f}",
                f"{speedup:.2f}x",
                status
            ])
    else:
        for B in BATCH_SIZES:
            baseline = baseline_results[B]
            table_data.append([
                B,
                f"{baseline['avg']:.4f}",
                f"{baseline['std']:.4f}"
            ])
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Summary
    if cuda_results is not None:
        print("\n" + "="*90)
        print(" SPEEDUP ANALYSIS")
        print("="*90)
        
        speedups = [baseline_results[B]['avg'] / cuda_results[B]['avg'] for B in BATCH_SIZES]
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)
        
        print(f"\n  Average speedup: {avg_speedup:.2f}x")
        print(f"  Min speedup: {min_speedup:.2f}x (batch size {BATCH_SIZES[np.argmin(speedups)]})")
        print(f"  Max speedup: {max_speedup:.2f}x (batch size {BATCH_SIZES[np.argmax(speedups)]})")
        
        # Check if target is met
        target_met = all(speedup >= 3.0 for speedup in speedups)
        if target_met:
            print(f"\n  🎉 Target achieved! All batch sizes have ≥3x speedup")
            return True
        else:
            failed_batches = [B for B, speedup in zip(BATCH_SIZES, speedups) if speedup < 3.0]
            print(f"\n  ⚠ Target not met. Failed batch sizes: {failed_batches}")
            return False
    else:
        print("\n  ⚠ CUDA kernel not available - only baseline results shown")
        return None


def save_results(baseline_results, cuda_results=None, filename="benchmark_results.txt"):
    """
    Save benchmark results to a file.
    
    Args:
        baseline_results (dict): Baseline benchmark results
        cuda_results (dict): CUDA benchmark results (optional)
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        f.write("="*90 + "\n")
        f.write(" FFN CUDA OPTIMIZATION BENCHMARK RESULTS\n")
        f.write("="*90 + "\n")
        f.write(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hidden size: {HIDDEN_SIZE}\n")
        f.write(f"Intermediate size: {INTERMEDIATE_SIZE}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"\n")
        
        if cuda_results is not None:
            headers = ["Batch", "Baseline(ms)", "CUDA(ms)", "Speedup"]
            table_data = []
            
            for B in BATCH_SIZES:
                baseline = baseline_results[B]
                cuda = cuda_results[B]
                speedup = baseline['avg'] / cuda['avg']
                table_data.append([
                    B,
                    f"{baseline['avg']:.4f}",
                    f"{cuda['avg']:.4f}",
                    f"{speedup:.2f}x"
                ])
            
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
            
            speedups = [baseline_results[B]['avg'] / cuda_results[B]['avg'] for B in BATCH_SIZES]
            f.write(f"Average speedup: {np.mean(speedups):.2f}x\n")
            f.write(f"Min speedup: {np.min(speedups):.2f}x\n")
            f.write(f"Max speedup: {np.max(speedups):.2f}x\n")
        else:
            headers = ["Batch", "Baseline(ms)"]
            table_data = [[B, f"{baseline_results[B]['avg']:.4f}"] for B in BATCH_SIZES]
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
            f.write("CUDA kernel not available\n")
    
    print(f"\n  Results saved to {filename}")


def run_benchmark():
    """Run all benchmarks and display results."""
    print("\n" + "="*90)
    print(" FFN CUDA OPTIMIZATION BENCHMARK")
    print("="*90)
    
    if not torch.cuda.is_available():
        print("\n  ⚠ CUDA not available. Running on CPU only.")
    
    # Benchmark baseline
    print("\n  Running baseline benchmark...")
    baseline_results = benchmark_baseline(BATCH_SIZES, WARMUP_ITERS, BENCHMARK_ITERS)
    print("  ✓ Baseline benchmark complete")
    
    # Benchmark CUDA kernel
    print("\n  Running CUDA kernel benchmark...")
    cuda_results = benchmark_cuda(BATCH_SIZES, WARMUP_ITERS, BENCHMARK_ITERS)
    if cuda_results is not None:
        print("  ✓ CUDA kernel benchmark complete")
    else:
        print("  ⚠ CUDA kernel not available")
    
    # Display results
    target_met = print_results(baseline_results, cuda_results)
    
    # Save results
    save_results(baseline_results, cuda_results)
    
    # Return exit code
    if cuda_results is None:
        return 0  # No CUDA kernel to test
    elif target_met:
        return 0  # Success
    else:
        return 1  # Failed to meet target


if __name__ == "__main__":
    exit_code = run_benchmark()
    sys.exit(exit_code)
