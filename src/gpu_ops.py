import cupy as cp
import numpy as np
import os

def transfer_to_gpu(A_host: np.ndarray, B_host: np.ndarray) -> tuple:
    """
    Transfers numpy arrays from Host (CPU) to Device (GPU).
    Args:
        A_host: First input matrix on host (CPU).
        B_host: Second input matrix on host (CPU).
    Returns:
        Two matrices A and B on device (GPU).
    """
    A_gpu = cp.asarray(A_host)
    B_gpu = cp.asarray(B_host)
    return A_gpu, B_gpu

def cupy_matmul_library(A_gpu: cp.ndarray, B_gpu: cp.ndarray) -> cp.ndarray:
    """
    Performs Matrix Multiplication using CuPy's optimized library.
    Args:
        A_gpu: First input matrix on device (GPU).
        B_gpu: Second input matrix on device (GPU).
    Returns:    
        The resulting matrix after multiplication C = A * B on device (GPU).
    """
    return cp.matmul(A_gpu, B_gpu)

def run_custom_kernel(kernel_source: str, function_name: str, grid: tuple, block: tuple, args: tuple):
    """
    Compiles and executes a raw CUDA kernel.
    Args:
        kernel_source: The source code of the CUDA kernel as a string.
        function_name: The name of the kernel function to execute.
        grid: The grid dimensions for kernel launch.
        block: The block dimensions for kernel launch.
        args: The arguments to pass to the kernel.
    """
    module = cp.RawModule(code=kernel_source)
    kernel = module.get_function(function_name)
    kernel(grid, block, args)