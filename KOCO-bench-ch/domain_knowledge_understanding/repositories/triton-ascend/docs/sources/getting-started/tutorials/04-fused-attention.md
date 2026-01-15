# 融合注意力 （Fused Attention）

在本节中，我们将编写一个比较复杂的 Tri Dao 的 Flash Attention v2 算法的 Triton 实现。

```Python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

DEVICE = "npu"

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  # 累加器、局部 l、局部 m、query 向量
                    K_block_ptr, V_block_ptr,  # 当前阶段的 key 和 value 块指针
                    start_m, qk_scale,  # 当前 query 块的起始位置，qk 缩放系数
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  # 块尺寸常量
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  # 当前阶段标志、m 和 n 的偏移索引
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr): # 上下文总长度，是否启用 FP8 的 value 精度
    # 设置当前阶段处理的 [lo, hi) 范围（以 column 块为单位）
    # causal = true
    # stage = 1
    # 因果注意力，顾名思义，它在计算时会限制信息的流动，只允许模型看到当前位置及之前的位置
    # 的信息。也就是说，当前位置的输出只能依赖于该位置及其之前的输入，而不能访问当前位置
    # 之后的信息。因果注意力保证了数据的顺序性，避免了“未来信息”的泄露。
    # 但是后面的逻辑也会触发
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M  # 第一阶段：处理 query 块之前的所有 token
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M # 第二阶段：处理当前 query 块本身
        lo = tl.multiple_of(lo, BLOCK_M)  # 对齐起始位置
    # causal = False （不需要 mask）
    else:
        lo, hi = 0, N_CTX # 处理全部上下文
        
    # 调整 K 和 V 块的指针位置到 lo 起始位置
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0)) # K 是 [HEAD_DIM, N_CTX]，第二维偏移 lo
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0)) # V 是 [N_CTX, HEAD_DIM]，第一维偏移 lo
    
    # 遍历当前阶段所有的 k、v 块并累加输出
    for start_n in range(lo, hi, BLOCK_N):  # 每次处理 BLOCK_N 列
        start_n = tl.multiple_of(start_n, BLOCK_N) # 对齐列起始位置
        # -- 计算 qk ----
        k = tl.load(K_block_ptr)
        # 修改K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # 对 STAGE 2 做 causal mask 操作
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :]) # 构造上三角 mask
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6) # 不满足条件的位置设置为 -∞
            m_ij = tl.maximum(m_i, tl.max(qk, 1)) # 更新 m_ij = max(m_i, max(qk))
            qk -= m_ij[:, None]  # 减去最大值做 softmax 稳定化
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale) # 缩放后最大值
            qk = qk * qk_scale - m_ij[:, None] # 稳定化
        # softmax 权重 p = exp(qk)
        p = tl.math.exp2(qk) # 使用 base-2 的指数函数提升数值稳定性
        l_ij = tl.sum(p, 1) # softmax 分母（每行的权重和）
        # -- 更新 m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)  # 更新因子：旧的最大值与新的最大值的指数差
        l_i = l_i * alpha + l_ij # 更新 softmax 累积分母
        # -- 更新输出累加器 --
        acc = acc * alpha[:, None]  # 累加器也乘以 alpha（保持缩放一致）
        # update acc
        v = tl.load(V_block_ptr) # 加载对应的 V 块
        # 根据是否启用 FP8 转换 softmax 权重类型
        if fp8_v:
            p = p.to(tl.float8e5)  # 转为 FP8 格式（节省显存）
        else:
            p = p.to(tl.float16) 
        # -------------------------------
        acc = tl.dot(p, v, acc) # 用 softmax 权重乘 V 累加到 acc 上
        m_i = m_ij # 更新当前块的最大值
        # 推进 V 和 K 块指针到下一个 BLOCK_N 范围
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i  # 返回累加的输出 acc，softmax 分母 l_i，最大值 m_i

@triton.jit
def _attn_fwd(Q, K, V, M, Out, sm_scale,  
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr, 
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr, 
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr, 
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr, 
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr, 
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # 断言 BLOCK_N 不超过 HEAD_DIM，确保 block 中的 K 不超过每个 token 的维度
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    # 当前处理的 M 方向上的 block index
    start_m = tl.program_id(0)
    # 循环处理所有 batch 和 head（Z * H 组 attention）
    for off_hz in range(0,Z*H):
    
        # 计算当前的 batch 和 head 索引
        off_z = off_hz // H
        off_h = off_hz % H
        
        # 对应 batch 和 head 的数据起始偏移
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        # 构造 Q 的 block pointer，指向一个 (BLOCK_M, HEAD_DIM) 的小块
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        
        # 构造 V 的 block pointer，初始从 (0, 0) 开始，后续在 inner 函数中前进
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            # why block_n??
            block_shape=(BLOCK_N, HEAD_DIM),  # 注意 V 是以 BLOCK_N 的 token 数处理的
            order=(1, 0),
        )
        
        # 构造 K 的 block pointer
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        # 构造输出 Out 的 block pointer
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        
        # 构造 M 轴和 N 轴的 index 偏移
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        
        # 初始化 m_i 和 l_i 为 attention 归一化的中间状态
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        # sm_scale 是 softmax 缩放系数，乘上 log2(e) 以配合 exp2 的 softmax 实现
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # 读取当前 block 的 query（BLOCK_M x HEAD_DIM）
        q = tl.load(Q_block_ptr)
        
        # STAGE 是一个标志位控制是否执行 Stage1（off-band）和 Stage2（on-band）
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            # 非对角线部分的 attention 计算
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  
                                            start_m, qk_scale,  
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  
                                            4 - STAGE, # Stage=1 -> inner Stage=3, Stage=3 -> inner Stage=1
                                            offs_m, 
                                            offs_n, 
                                            N_CTX, 
                                            V.dtype.element_ty == tl.float8e5 # 是否使用 fp8
                                            )
            
        # stage 2: on-band
        if STAGE & 2:
            # 对角线块（on-band）部分的 attention 计算
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                            start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
        # epilogue：归一化 + log
        m_i += tl.math.log2(l_i) # 计算 softmax 输出的 logsumexp
        acc = acc / l_i[:, None] # 对 acc 做 softmax 归一化
        # 保存 logsumexp 到 M，保存最终输出到 Out
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}

        grid = (triton.cdiv(q.shape[2], BM),1, 1)
        # (1, 2, 1024)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        # 调用 Triton 内核（注意 stride 和 block 尺寸都必须显式传入）
        _attn_fwd[grid](
            q, k, v, M, o, sm_scale, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], 
            N_CTX=q.shape[2], # 序列长度
            HEAD_DIM=HEAD_DIM_K,  # 64
            BLOCK_M = BM, # 32
            BLOCK_N = BN, # 32
            STAGE=stage,
            debug=True,
            **extra_kern_args)
        # 保存上下文以供 backward 使用
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

attention = _attention.apply

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal, dtype, BM , BN",[
    (4, 32, 32, 64, False, torch.float16, 32, 32),
    (4, 32, 64, 64, False, torch.float16, 32, 64),
    (1, 2, 128, 128, False, torch.float16, 32, 128),
    (1, 1, 128, 128, False, torch.float16, 64, 128),
    (1, 1, 128, 128, False, torch.float16, 32, 128),
    (2, 2, 128, 256, False, torch.float16, 32, 128),
    (1, 2, 256, 256, False, torch.float16, 32, 256),
])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN ):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
  
    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    tri_out = attention(q, k, v, causal, sm_scale,BM,BN ).half()
    
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)
    print(f"Attention BM: {BM}, BN: {BN}, dtype: {dtype} PASSED!")
   
if __name__ == "__main__":
   test_op(4,32,32,64, causal=False, dtype=torch.float16, BM = 32,BN = 32)
   test_op(4,32,64,64, causal=False, dtype=torch.float16, BM = 32,BN = 64)
   test_op(1,2,128,128, causal=False, dtype=torch.float16, BM = 32,BN = 128)
   test_op(1,1,128,128, causal=False, dtype=torch.float16, BM = 64,BN = 128)
   test_op(1,1,128,128, causal=False, dtype=torch.float16, BM = 32,BN = 128)
   test_op(2,2,128,256, causal=False, dtype=torch.float16, BM = 32,BN = 128)
   test_op(1,2,256,256, causal=False, dtype=torch.float16, BM = 32,BN = 256)
```

Out:

```bash
Attention BM: 32, BN: 32, dtype: torch.float16 PASSED!
Attention BM: 32, BN: 64, dtype: torch.float16 PASSED!
Attention BM: 32, BN:128, dtype: torch.float16 PASSED!
Attention BM: 64, BN: 128, dtype: torch.float16 PASSED!
Attention BM: 32, BN: 128, dtype: torch.float16 PASSED!
Attention BM: 32, BN: 128, dtype: torch.float16 PASSED!
Attention BM: 32, BN: 256, dtype: torch.float16 PASSED!
```

上面输出日志表明Triton和PyTorch上的输出结果完全一致。
