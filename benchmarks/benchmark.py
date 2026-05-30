import torch
import triton
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.triton_attention import triton_attention
from src.torch_attention import torch_attention

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N_CTX'],  
        x_vals=[1024, 2048, 4096, 8192],  
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch Attention', 'Triton Kernel'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Execution Time (ms)',
        plot_name='attention-performance',
        args={'BATCH': 2, 'HEADS': 4, 'HEAD_DIM': 64}
    )
)
def benchmark(BATCH, HEADS, N_CTX, HEAD_DIM, provider):
    q = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), device='cuda', dtype=torch.float16)
    k = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), device='cuda', dtype=torch.float16)
    v = torch.randn((BATCH, HEADS, N_CTX, HEAD_DIM), device='cuda', dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_attention(q, k, v), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_attention(q, k, v), quantiles=quantiles)

    # Free memory instantly to protect the 4GB GTX 1650
    del q, k, v
    torch.cuda.empty_cache()
    
    return ms, min_ms, max_ms

if __name__ == '__main__':
    print("Executing GPU Benchmark Harness...")
    benchmark.run(save_path='.', print_data=True)