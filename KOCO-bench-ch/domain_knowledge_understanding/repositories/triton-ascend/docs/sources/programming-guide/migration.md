# 迁移指南

在将Triton内核从GPU迁移到NPU的过程中，开发者通常会遇到一些特有的技术挑战。本章节由基本迁移步骤、常见问题概览、解决 coreDim 超限问题、
处理复合问题：coreDim + UB 溢出、IR转换不兼容五部分构成，以帮助您顺利完成迁移工作。

# 0 基本迁移步骤

首先需要了解从 GPU 迁移到 NPU 的基本步骤。以下是一个可在 GPU 上正常运行的 Triton 内核示例：

```python
import pytest
import torch
import triton
import triton.language as tl

@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr, XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)

@pytest.mark.parametrize('shape', [(1,), (2,), (4,)])
@pytest.mark.parametrize('dtype', [torch.int32])
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4

    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='cuda')
    std = torch.broadcast_to(x, (YS, XS))
    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='cuda')
    fn_broadcast_1d[(1,)](output, x, XS, YS)
    assert torch.allclose(std, output)
```

迁移到 NPU 的第一步，只需将 device='cuda' 改为 device='npu'，即可尝试在 NPU 上运行：

```python
import pytest
import torch
import triton
import triton.language as tl

@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr, XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)

@pytest.mark.parametrize('shape', [(1,), (2,), (4,)])
@pytest.mark.parametrize('dtype', [torch.int32])
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4

    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='npu')
    std = torch.broadcast_to(x, (YS, XS))
    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='npu')
    fn_broadcast_1d[(1,)](output, x, XS, YS)
    assert torch.allclose(std, output)
```

# 1 常见问题概览

完成迁移基础步骤后，可能会遇到新的问题，新问题可归纳为以下三类：

1. **coreDim限制问题**  
   当网格维度超过NPU硬件限制时触发。  
   典型错误信息：`coreDim=xxxx can't be greater than UINT16_MAX`

2. **UB空间溢出**  
   内存使用超出NPU缓存容量。  
   典型错误信息：`ub overflow, requires xxxx bits while 1572684 bits available!`

3. **IR转换不兼容**  
   某些GPU特有的操作在NPU上不被支持。

接下来我们将通过具体示例来详细说明前两类问题的解决方法。

# 2 解决 coreDim 超限问题

## 问题分析

NPU的 `coreDim` 参数不能超过 `UINT16_MAX`（65535）。当处理大规模数据时，简单的grid划分可能导致该限制被突破。

## 案例：`zeros_like` 函数优化

**问题场景：**
- 数据规模：`N = 1,073,741,824`
- 原始 `BLOCK_SIZE = 2048`
- 计算得到的 `coreDim = 524,288 > 65535`（超限）

**解决思路1：**  
昇腾编译器针对coreDim超限问题，有对应的解决方案，只需将环境变量'TRITON_ALL_BLOCKS_PARALLEL'和'ENABLE_UNPUBLISHED_FEATURE'设为1。设置命令如下：

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
export ENABLE_UNPUBLISHED_FEATURE=1
```

**解决思路2：**  
通过增大 `BLOCK_SIZE` 来减少所需的核心数量，确保 `coreDim` 不超过限制。

***计算公式：***
coreDim = ceil(N / BLOCK_SIZE)
=> 需满足：ceil(N / BLOCK_SIZE) <= 65535
=> BLOCK_SIZE >= ceil(N / 65535)
代入 N = 1,073,741,824 得：
BLOCK_SIZE >= triton.next_power_of_2(triton.cdiv(1073741824, 65535)) = 32768 → 至少为 32768 更稳妥

优化前的代码：
```Python
import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype
    
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    
    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # 原始值过小
    return out
```
优化后的代码：
```Python
import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype
    
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    
    # 动态计算适合的 BLOCK_SIZE 以避免 coreDim 超限
    optimal_block_size = 32768  # 根据计算得出的优化值
    grid_fn = lambda meta: (triton.cdiv(N, optimal_block_size),)
    
    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=optimal_block_size)
    return out
```

# 3 处理复合问题：coreDim + UB 溢出

## 问题分析

在某些情况下，解决了 coreDim 问题后可能引发新的UB溢出问题。这通常发生在增大 BLOCK_SIZE 后，单个线程块需要处理的数据量超出了NPU的UB缓存容量。

## 案例

**问题场景：**

- 数据规模：`N = 1,073,741,824`
- 原始 `BLOCK_SIZE = 4096`
- 计算得到的 `coreDim = 262,144 > 65535`（超限）
- 调整为 BLOCK_SIZE = 32768 后，coreDim = 32,768（合规）
- 但出现 UB 溢出

**解决思路：**

引入 BLOCK_SIZE_SUB 参数，将大块进一步细分，在保持合理 coreDim 的同时控制内存使用。

优化前代码：
```Python
import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)
    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)


def masked_fill(inp, mask, value):
    # ... 参数验证代码 ...
    # inp.device = "npu"
    N = inp.numel()
    if N == 0:
        return out
    
    grid = lambda meta: (triton.cdiv(N, 4096),)  # 导致 coreDim 超限
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N, 4096)
    return out
```

优化后代码：
```Python
import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, 
                      BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # 计算需要处理的子块数量
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    # 分块处理，避免 UB 溢出
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # 分批加载和处理数据
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # 先写入原始数据
        tl.store(out + offsets, input_vals, mask=mask)

        # 然后在需要填充的位置覆写目标值
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)


def masked_fill(inp, mask, value):
    logger.debug("GEMS MASKED FILL")
    
    # ... 参数验证代码 ...
    # inp.device = "npu"
    N = inp.numel()
    if N == 0:
        return out
    
    # 使用优化的参数配置
    MAIN_BLOCK_SIZE = 32768  # 确保 coreDim 合规
    SUB_BLOCK_SIZE = 1024    # 控制 UB 使用量
    
    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N, 
                           MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

# 4 IR转换不兼容

对于triton-ascend不支持IR转换场景，可以重写kernel规避该场景，也可以提issue来解决。