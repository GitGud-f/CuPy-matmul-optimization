"""
Module: cpu_baseline.py
Description:
    CPU Baseline Implementation for Matrix Multiplication.
    This module provides a straightforward CPU implementation of matrix multiplication
    using triple nested loops.
Functions:
    - cpu_matmul: Performs matrix multiplication on two input matrices A and B.
"""

import numpy as np

def cpu_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Standard Matrix Multiplication using Triple Nested Loops.
    C[i][j] = sum(A[i][k] * B[k][j])
    Args:
        A: First input matrix.
        B: Second input matrix.
    Returns:
        The resulting matrix after multiplication C = A * B.
    """

    A = np.array(A)
    B = np.array(B)
    
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    
    if cols_A != rows_B:
        raise ValueError("Cannot multiply: Dimensions do not match.")
        
    C = np.zeros((rows_A, cols_B), dtype=A.dtype)
    
    for i in range(rows_A):          
        for j in range(cols_B):      
            total = 0
            for k in range(cols_A):  
                total += A[i, k] * B[k, j]
            C[i, j] = total
            
    return C