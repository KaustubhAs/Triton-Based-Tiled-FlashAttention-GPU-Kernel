import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    sm_scale,  # <--- Added precomputed scale factor
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """
    Triton kernel for fused Scaled Dot-Product Attention.
    Computes Softmax(Q @ K^T * sm_scale) @ V using SRAM tiling.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + k_offset + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX - start_n, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX - start_n, other=0.0)
        
        # Compute Q @ K^T
        qk = tl.dot(q, tl.trans(k))
        
        # Apply the precomputed scaling factor
        qk = qk * sm_scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        acc += tl.dot(p.to(tl.float16), v)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    
    out_ptrs = Out + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

def triton_attention(q, k, v):
    BATCH, HEADS, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, 64), BATCH * HEADS, 1)
    
    # Calculate scale exactly once on the CPU
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    _fwd_kernel[grid](
        q, k, v, out,
        sm_scale,  # <--- Pass the float here
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BATCH, HEADS, N_CTX,
        BLOCK_M=64, BLOCK_N=64, HEAD_DIM=HEAD_DIM
    )
    return out