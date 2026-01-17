# Parallel Matrix Multiplication with CUDA & CuPy

---

## Project Overview

This repository implements **Matrix Multiplication ($C = A \times B$)** using various optimization levels to demonstrate the power of GPU computing. The project evolves from a naive CPU implementation to a highly optimized Tiled CUDA kernel using Shared Memory.

The implementation is written in **Python** using **NumPy** for the baseline and **CuPy** for GPU operations. Custom CUDA kernels are written in **C++** and compiled dynamically at runtime.

### Objectives
1.  **CPU Baseline:** Establish a reference time using Python loops.
2.  **GPU Library:** Utilize `cupy.matmul` (cuBLAS) for peak performance comparison.
3.  **Naive Kernel:** Implement a custom CUDA kernel using Global Memory.
4.  **Tiled Kernel:** Implement **Shared Memory Tiling** to reduce memory bandwidth bottlenecks.

---

## Repository Structure

The project is organized to separate source code, kernels, and analysis logic.

```text
CuPy-matmul-optimization/
├── kernels/                 # CUDA C++ Source Files
│   ├── matmul.cu            # Basic global memory implementation
│   └── tiled_matmul.cu      # Optimized Shared Memory implementation
├── notebooks/               # Execution Environment
│   └── benchmark_analysis.ipynb  # Main Lab Report & Experiments
├── src/                     # Python Source Code
│   ├── __init__.py      
│   ├── cpu_baseline.py      # Naive CPU implementation
│   ├── gpu_ops.py           # CuPy wrappers & Kernel loaders
│   └── utils.py             # Scientific benchmarking & verification tools
├── .gitignore        
├── License 
├── README.md                # Documentation      
└── requirements.txt
```

---

## Usage

### 1. Environment Setup
This project requires a GPU environment. **Google Colab** is recommended.
If not using colab, virtual env is recommended

- Make virtual env
``` bash
python -m venv venv  
```
- activate venv
``` bash
source ./venv/bin/activate 
```
- install requirements
```bash
pip install -r requirements.txt
```


### 2. Running the Benchmarks
The primary entry point is the Jupyter Notebook.
1.  Open `notebooks/benchmark_analysis.ipynb`.
2.  Run the cells sequentially. (If not on colab thee files starting with %%writefile should be ignored)
3.  The notebook handles:
    *   Generating random matrices ($N \times N$).
    *   Compiling `.cu` kernels on the fly.
    *   Validating results against NumPy.
    *   Comparing results

---

## 📊 Benchmark Results

Performance measured on **NVIDIA T4 (Google Colab)** with Matrix Size $N=2000$.

| Implementation | Configuration | Time (ms) | Speedup vs Naive |
| :--- | :--- | :--- | :--- |
| **Naive Kernel** | Block 16x16 | 39.66 ms | 1.00x (Baseline) |
| **Tiled Kernel** | Block 16x16 | 26.46 ms | 1.50x |
| **Tiled Kernel** | **Block 32x32** | **21.72 ms** | **1.83x** |
| **CuPy Library** | cuBLAS | 3.48 ms | 11.41x |

> **Analysis:** Implementing Tiling with Shared Memory reduced execution time by **~57%** compared to the Naive approach.

---

## Theoretical Concepts (Q&A)

### Why use 2D Indexing?
**Formula:** `row = blockIdx.y * blockDim.y + threadIdx.y;`

While GPU memory is linear (1D), matrices are logical 2D structures.
1.  **Logical Mapping:** This formula maps the grid of threads directly to matrix coordinates $(i, j)$.
2.  **Coalesced Access:** By mapping `threadIdx.x` to the columns, consecutive threads in a Warp access consecutive memory addresses. This allows the memory controller to merge 32 reads into a single transaction (Coalescing), which is critical for performance.

### How does Tiling improve performance?
The **Naive Kernel** is bandwidth-bound. Every thread reads rows of A and columns of B from **Global Memory** (DRAM), which has high latency.

**Tiling** solves this by:
1.  Loading a small block ($T \times T$) of data into **Shared Memory** (L1 Cache).
2.  Synchronizing threads (`__syncthreads()`).
3.  Computing partial products using the fast Shared Memory data.
4.  This reduces Global Memory accesses by a factor of $T$ (Tile Width).

---

## Dependencies
*   Python 3.8+
*   NumPy
*   CuPy (matches your CUDA version)

---