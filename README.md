# Fused Convolution Kernel Optimization (CUDA)

## Overview
This repository contains a highly optimized forward convolution kernel for Convolutional Neural Networks (CNNs), implemented in CUDA C++. It was developed for the UIUC ECE 408 (Applied Parallel Programming) CNN Performance Competition. 

The core of this project is a custom **fused unroll-matmul-permute** kernel that significantly accelerates CNN inference by minimizing global memory accesses and maximizing GPU parallel compute capabilities.

## Results & Performance
* **Rank:** 5th out of 122 students on the course leaderboard.
* **Total Runtime:** 19.42 ms (Layer 1: 11.45 ms | Layer 2: 7.96 ms).

## Technical Implementation
The custom kernel (`Project_CNN/project/src/layer/custom/m3-forward.cu`) replaces the standard nested-loop convolution with a tiled matrix multiplication (GEMM) approach, featuring the following optimizations:

* **Kernel Fusion:** Fused the `im2col` (unroll), matrix multiplication, and output permutation steps into a single CUDA kernel. This avoids writing the massive intermediate `im2col` matrix to global memory, strictly bounding memory bandwidth limitations.
* **Shared Memory Tiling:** The convolution filter weights (masks) are loaded into `__shared__` memory blocks to broadcast data efficiently across multiple threads within a block, reducing redundant global memory fetches.
* **Register Tiling:** Input feature maps are cached directly in thread-local registers (`N_reg`), enabling extremely fast access during the inner product accumulations.
* **Loop Unrolling:** Applied `#pragma unroll` to the inner matrix-multiplication loops to reduce branch overhead and increase instruction-level parallelism.
* **Grid & Block Tuning:** Utilizes a 1D block layout (`dim3 blockDim(64, 1, 1)`) mapping to N-dimension outputs, paired with 2D grids mapping out the M and N matrix dimensions to maximize Streaming Multiprocessor (SM) occupancy.
