import time
import numpy as np

def generate_matrices(n, dtype=np.float32):
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

def check_correctness(target, reference, tolerance=1e-4):
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

def benchmark_function(func, name, *args):
    """
    Benchmarks the execution time of a given function.
    Args:
        func: The function to benchmark.
        name: Name of the function (for reporting).
        *args: Arguments to pass to the function.
    Returns:
        A tuple containing the result of the function and the execution time in milliseconds.
    """
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    
    execution_time_ms = (end_time - start_time) * 1000
    print(f"[{name}] Execution Time: {execution_time_ms:.4f} ms")
    return result, execution_time_ms