"""
Module: utils.py

Description: 
    Utility functions for matrix generation, correctness checking, and benchmarking.
Functions:
    - generate_matrices: Generates two random N*N matrices.
    - check_correctness: Compares two matrices for correctness.
    - benchmark_function: Benchmarks a given function by measuring execution time.
"""
from typing import Tuple
import time
import numpy as np
import cupy as cp

def generate_matrices(n: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates two random N*N matrices A and B.
    Using float32 is standard for GPU programming (single precision).
    Args:
        n: Size of the matrices (N x N).
        dtype: Data type of the matrices (default: np.float32).
    Returns: 
        Two N x N matrices A and B.
    """
    
    A = np.random.rand(n, n).astype(dtype)
    B = np.random.rand(n, n).astype(dtype)
    return A, B

def check_correctness(target: np.ndarray, reference: np.ndarray, tolerance: float = 1e-4) -> bool:
    """
    Compares two matrices using NumPy's allclose.
    Args:
        target: The matrix to test.
        reference: The reference matrix.
        tolerance: The tolerance for comparison (default: 1e-4).
    Returns:
        True if matrices are close within the given tolerance, False otherwise.
    """
    
    if hasattr(target, 'get'): 
        target = target.get()
    if hasattr(reference, 'get'): 
        reference = reference.get()
    try:
        np.testing.assert_allclose(target, reference, atol=tolerance, rtol=tolerance)
        return True
    except AssertionError:
        return False

def benchmark_function(func, name: str, *args, n_iter: int = 5) -> Tuple[np.ndarray, float]:
    """
    Benchmarks a given function by running it multiple times and measuring execution time.
    Args:
        func: The function to benchmark.
        name: Name of the benchmark (for reporting).
        *args: Arguments to pass to the function.
        n_iter: Number of iterations to run (default: 5).
    Returns:
        A tuple containing the result of the function and the average execution time in milliseconds.
    """
    func(*args) # warm-up
    cp.cuda.Device(0).synchronize()
    
    timings = []
    result = None

    for i in range(n_iter):
        cp.cuda.Device(0).synchronize()
        
        start_time = time.perf_counter()
        result = func(*args) 
        
        cp.cuda.Device(0).synchronize()
        end_time = time.perf_counter()
        
        timings.append((end_time - start_time) * 1000)

    avg_time = np.mean(timings)
    std_dev = np.std(timings)
    
    print(f"[{name}] Avg Time: {avg_time:.4f} ms (±{std_dev:.2f} ms) | Runs: {n_iter}")
    
    return result, avg_time