# Triton-Based Tiled FlashAttention GPU Kernel

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2205.14135-B31B1B.svg)](https://arxiv.org/pdf/2205.14135)
[![Framework](https://img.shields.io/badge/Language-Triton-blue)](https://triton-lang.org/main/index.html)
[![Backend](https://img.shields.io/badge/Backend-PyTorch_CUDA-EE4C2C)](https://pytorch.org/)

An end-to-end, hardware-optimized implementation of a fused, tiled FlashAttention-style GPU kernel written in OpenAI Triton. This project demonstrates low-level GPU memory hierarchy utilization, custom kernel compilation, and performance benchmarking on consumer-grade hardware under highly restricted VRAM budgets.

---

## Technical Overview

Standard transformers execute Scaled Dot-Product Attention using the classical formula:

$$Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In standard frameworks like PyTorch, this calculation is **memory-bound**. The intermediate attention matrix of shape `[Sequence Length, Sequence Length]` ($O(N^2)$ memory footprint) is written to and read from slow High-Bandwidth Memory (HBM) multiple times. This introduces severe memory bottlenecks and frequently triggers Out-Of-Memory (OOM) failures on low-VRAM hardware.

This project implements a fused GPU kernel that bypasses HBM round-trips using **SRAM Tiling**. By breaking the Query, Key, and Value matrices into small `64x64` blocks, the entire calculation is executed locally inside the GPU's ultra-fast Streaming Multiprocessor (SM) Shared Memory (SRAM), dropping the memory traffic complexity down to $O(N)$.

---

## Hardware Environment Constraints

This project was built, compiled, and benchmarked locally under a $0 cloud budget on highly constrained consumer hardware:

* **Operating System:** Windows Subsystem for Linux (WSL2 - Ubuntu 22.04 LTS)
* **Host RAM:** 8GB DDR4 (7.78GB usable)
* **GPU Architecture:** NVIDIA GeForce GTX 1650 (Turing Architecture, Compute Capability 7.5)
* **VRAM Capacity:** 4GB GDDR5
* **CUDA Driver Version:** 13.2 (Runtime 12.1)

---

## Algorithmic & Low-Level Optimizations

1. **Static Scale Projection:** To minimize redundant mathematical instructions inside the inner hardware loops, the softmax scaling factor ($\frac{1}{\sqrt{d_k}}$) is pre-computed on the CPU host and passed directly into the GPU registers, saving valuable arithmetic execution cycles.
2. **Online Softmax Integration:** To evaluate Softmax block-by-block without access to global context, the kernel implements a numerically stable online softmax scaling calculation, updating running maximums ($m_i$) and running denominators ($l_i$) purely within SRAM registers.
3. **Hardware Tiling Bounds:** Block sizes (`BLOCK_M=64`, `BLOCK_N=64`) are explicitly constrained to align cleanly with the 64KB Shared Memory capacity per SM of the GTX 1650, ensuring maximum hardware occupancy without triggering register spilling.

---

## Performance Results & Empirical Benchmarks

The benchmark evaluated the performance of standard PyTorch Attention against the custom Triton Kernel under a fixed workload (`Batch Size = 2`, `Heads = 4`, `Head Dimension = 64`) across expanding sequence lengths using half-precision (`FP16`).

### Execution Latency Comparison

| Sequence Length (`N_CTX`) | PyTorch Attention (ms) | Custom Triton Kernel (ms) | Speedup Multiplier |
| :--- | :--- | :--- | :--- |
| **1024** | 9.12 ms | 5.29 ms | **1.72x Faster** |
| **2048** | 38.69 ms | 20.06 ms | **1.93x Faster** |
| **4096** | 145.00 ms | 78.30 ms | **1.85x Faster** |
| **8192** | 903.64 ms | 509.57 ms | **1.77x Faster** |

### Key Architectural Takeaways:
* **Sustained ~1.8x Speedup:** The Triton kernel consistently outperforms PyTorch by optimizing data reuse in local registers and eliminating global memory traffic.
* **OOM Resilience:** While standard PyTorch attention approaches memory exhaustion as sequence lengths expand due to $O(N^2)$ tensor materialization, the Triton kernel maintains a strict linear memory growth curve ($O(N)$), cleanly executing an 8,192 token sequence context within a strict 4GB VRAM limit.

---

## Project Structure

```text
triton-flash-attention/
├── src/
│   ├── __init__.py
│   ├── triton_attention.py   # Triton GPU compiled kernel & wrapper
│   └── torch_attention.py    # PyTorch baseline implementation
├── benchmarks/
│   ├── __init__.py
│   └── benchmark.py          # Latency testing harness with explicit CUDA caching
├── requirements.txt
└── README.md
```

## Setup & Execution
Prerequisites
Ensure you are running inside a WSL2 Ubuntu environment with the NVIDIA Container Toolkit active.

```Bash
# Clone the repository and navigate inside
cd triton-flash-attention

# Install C++ compiler toolchain required for Triton JIT compilation
sudo apt-get update && sudo apt-get install build-essential -y

# Initialize python environment
python3 -m venv triton_env
source triton_env/bin/activate

# Install exact dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt
```

Running the Benchmarks
To execute the compiler and run the latency benchmarking harness:

```Bash
python benchmarks/benchmark.py
```

## References
- Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135
- OpenAI Triton Language Documentation: https://triton-lang.org/