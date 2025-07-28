import triton
import triton.language as tl
import torch
# 矩阵a 矩阵b 矩阵bias 矩阵c

@triton.jit
def linear_kernel(  # 与 matmul_kernel 基本一致，只是多了 bias
        x_ptr, w_ptr, b_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,          # x : (M, K)
        stride_wk, stride_wn,          # *** w 被当成 Wᵀ(K, N) 来读 ***
        stride_ym, stride_yn,          # y : (M, N),                      # b : (N,)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
        HAS_BIAS: tl.constexpr
):
    # ---------------- 1.  Block mapping ----------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---------------- 2.  生成 A、B 块的指针 ----------------
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k  = tl.arange(0, BLOCK_SIZE_K)

    
    a_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    # ---------------- 3.  主循环 ----------------
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k*BLOCK_SIZE_K, other=0.)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k*BLOCK_SIZE_K, other=0.)
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_wk

    # ---------------- 4.  可选激活 ----------------
    if ACTIVATION == "leaky_relu":
        acc = tl.where(acc > 0, acc, 0.01 * acc)

    # ---------------- 5.  加 bias ----------------
    if HAS_BIAS:
        bias_ptrs = b_ptr + offs_bn
        bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.).to(tl.float32)
        acc += bias[None, :]

    y_block = acc.to(tl.float16)


    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y_block, mask=mask)
    
    

def triton_linear(
    x: torch.Tensor,                  # (M, K)
    weight: torch.Tensor,             # (N, K)  注意：已转置！
    bias: torch.Tensor,               # (N,)
    activation: str = None,           # "leaky_relu" | "relu" | None
):
    assert x.dim() == 2, "x must be 2-D"
    assert weight.dim() == 2, "weight must be 2-D"
    assert bias.dim() == 1, "bias must be 1-D"
    assert x.shape[1] == weight.shape[1], "in_features mismatch"
    assert weight.shape[0] == bias.shape[0], "bias length mismatch"

    M, K = x.shape
    N, _ = weight.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 64
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) *
                triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    linear_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # key here
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=8,
        ACTIVATION=activation or ""
    )
    return y


def triton_linear_3d(
    x: torch.Tensor,                  # (M, K)
    weight: torch.Tensor,             # (N, K)  注意：已转置！
    bias: torch.Tensor,               # (N,)
    activation: str = None,           # "leaky_relu" | "relu" | None
):
    assert x.dim() == 3 or x.dim() == 2, "x must be 2-D or 3-D"
    B, S, H_in = x.shape
    H_out = weight.shape[0]
    x_2d = x.view(-1, H_in) 
    y_2d = torch.empty(B*S,H_out, device=x.device, dtype=torch.float16)
    
  
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    def grid(META):
        return (triton.cdiv(B*S, META["BLOCK_SIZE_M"]) *
                triton.cdiv(H_out, META["BLOCK_SIZE_N"]),)
    
    linear_kernel[grid](
        x_2d, weight, bias, y_2d,
        B * S, H_out, H_in,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),  # key here
        y_2d.stride(0), y_2d.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=4,
        ACTIVATION=activation or "",
        HAS_BIAS=(bias is not None)
    )

    return y_2d.view(B, S, H_out)

