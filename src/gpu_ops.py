"""
Module: gpu_ops.py

Description: 
    This module contains functions to handle GPU operations using CuPy,
    including data transfer, matrix multiplication using CuPy's library, and executing
    custom CUDA kernels.
    
Functions:
    - transfer_to_gpu: Transfers numpy arrays from Host (CPU) to Device (GPU).
    - cupy_matmul_library: Performs Matrix Multiplication using CuPy's optimized library.
    - run_custom_kernel: Compiles and executes a raw CUDA kernel.
"""

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

def run_custom_kernel(A_gpu: cp.ndarray, B_gpu: cp.ndarray, N: int, block_size=(16, 16)) -> cp.ndarray:
    """
    Runs a custom naive matrix multiplication kernel.
    Args:
        A_gpu: First input matrix on device (GPU).
        B_gpu: Second input matrix on device (GPU).
        N: Size of the NxN matrices.    
        block_size: The block size to use in the kernel launch (default: (16,16)).
    Returns:    
        The resulting matrix C = A * B on device (GPU).
    """

    with open('kernels/matmul.cu', 'r') as f:
        kernel_code = f.read()
    
    kernel = cp.RawKernel(kernel_code, 'matmul_kernel')
    
    C_gpu = cp.zeros((N, N), dtype=cp.float32)
    
    grid_x = (N + block_size[0] - 1) // block_size[0]
    grid_y = (N + block_size[1] - 1) // block_size[1]
    grid_dim = (grid_x, grid_y)
    
    kernel(grid_dim, block_size, (A_gpu, B_gpu, C_gpu, cp.int32(N)))
    
    return C_gpu

def run_tiled_kernel_dynamic(A_gpu: cp.ndarray, B_gpu: cp.ndarray, N: int, tile_size: int =16) -> cp.ndarray:
    """
    Runs a tiled matrix multiplication kernel with dynamic tile size.
    Args:
        A_gpu: First input matrix on device (GPU).
        B_gpu: Second input matrix on device (GPU).
        N: Size of the NxN matrices.
        tile_size: The tile size to use in the kernel.
    Returns:    
        The resulting matrix C = A * B on device (GPU).
    """

    with open('kernels/tiled_matmul.cu', 'r') as f:
        raw_code = f.read()
    

    augmented_code = f"#define TILE_WIDTH {tile_size}\n" + raw_code
    
    kernel = cp.RawKernel(augmented_code, 'tiled_matmul_kernel')
    
    C_gpu = cp.zeros((N, N), dtype=cp.float32)
    

    block_dim = (tile_size, tile_size)
    grid_x = (N + tile_size - 1) // tile_size
    grid_y = (N + tile_size - 1) // tile_size
    
    kernel((grid_x, grid_y), block_dim, (A_gpu, B_gpu, C_gpu, cp.int32(N)))
    
    return C_gpu