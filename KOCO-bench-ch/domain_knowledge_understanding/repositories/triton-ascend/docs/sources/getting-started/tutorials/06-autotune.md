# 自动调优 （Autotune）

在本节中，我们将展示使用 Triton 的 autotune 方法以自动选择最优的 kernel 配置参数。当前 Triton-Ascend autotune 完全兼容社区 autotune 的使用方法（参考[社区文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)），即需要用户手动传入一些定义好的 triton.Config，然后 autotune 会通过 benchmark 的方式选择其中的最优 kernel 配置；此外 Triton-Ascend 提供了**进阶的 autotune** 用法，用户需要提供当前 triton kernel 的切分轴、tiling 轴等信息，此时 autotune 会根据实际的输入大小自动生成一些可能最优的 kernel 配置，然后通过 benchmark 或者 profiling 的方式选择其中的最优配置。

说明：
当前Triton-Ascend autotune支持block size、multibuffer（编译器的优化），因为硬件架构差异不支持num_warps、num_statges参数，未来还会持续增加autotune可调项。

## 社区 autotune 使用示例
```Python
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():

    # 返回一组不同的 kernel 配置，用于 autotune 测试
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]

    @triton.autotune(
        configs=get_autotune_config(),      # 配置列表
        key=["numel"],                      # 当numel大小发生变化时会触发autotune
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # 块大小，用于控制每个线程块处理多少数据
    ):
        pid = tl.program_id(0)            # 获取当前 program 的 ID
        idx = pid * XS + tl.arange(0, XS) # 当前线程块处理的 index 范围
        msk = idx < numel                 # 避免越界的掩码

        # 重复执行一些计算以模拟负载（并测试性能）/ Repeat computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # 加载 x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # 加载 x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # 计算
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # 存储到输出

    # Triton 调用函数，自动使用 autotuned kernel
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # 计算 grid 大小 
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 使用 PyTorch 作为参考实现进行对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"                         # 使用 NPU 作为设备
    DTYPE = torch.float32
    N = 192 * 1024                      # 输入长度
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x0
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x1
    torch_ref = torch_calc_func(x0, x1)              # 得到参考结果
    triton_cal = triton_calc_func(x0, x1)            # 运行 Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # 验证输出是否一致

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # 输出成功标志 / Print success message
```

## 进阶 autotune 使用示例
```Python
# 下面是相对于社区autotune所增加的参数和类型有所修改的参数
# 注意：当split_params和tiling_params有一个参数不为空时即会自动触发进阶的autotune调优方法

# key (Dict[str, str]): axis name: argument name组成的字典，argument 变化会触发候选配置的重新生成与评估
#     axis name 属于集合 {'x','y','z','w','v','t','rx','ry','rz','rw','rv','rt}，前缀 'r' 表示规约轴
#     只有此参数中的轴名称在作为规约轴时才应该添加前缀 r
# split_params (Dict[str, str]): axis name: argument name组成的字典, argument 是切分轴的可调参数, 例如 'XBLOCK'
#     axis name必须在参数key的轴名称集合里。 请勿在轴名称前添加前缀 r
#     此参数可以为空，当split_params 和 tiling_params 都为空的时候不会进行自动寻优
#     切分轴通常可以根据 `tl.program_id()` 分核语句来确定
# tiling_params (Dict[str, str]): axis name: argument name组成的字典， argument 是分块轴的可调参数, 例如 'XBLOCK_SUB'
#     axis name必须在参数key的轴名称集合里。请勿在轴名称前添加前缀 r
#     此参数可以为空，当split_params 和 tiling_params 都为空的时候不会进行自动寻优
#     分块轴通常可以根据 `tl.arange()` 分块表达式来确定
# low_dims (List[str]): 所有低维轴的轴名称列表，axis name必须在参数key的轴名称集合里， 请勿在轴名称前添加前缀 r
# dual_reduction (bool): 是否在多个轴上做规约，会影响tiling生成策略
# persistent_reduction (bool): 是否在规约轴上是否做切分，会影响tiling生成策略
# 使用可以参考 ascend\examples\autotune_cases 里面的案例
@triton.autotune(
    configs=[],
    key={"x": "n_elements"},           # 切分轴x对应的大小
    split_params={"x": "BLOCK_SIZE"},  # 切分轴x需要调整的BLOCK_SIZE大小
    tiling_params={},                  # tiling轴即切分轴
    low_dims=["x"],                    # 低维轴
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def add_kernel(
    x_ptr,  # *指向*第一个输入向量的指针。
    y_ptr,  # *指向*第二个输入向量的指针。
    output_ptr,  # *指向*输出向量的指针。
    n_elements,  # 向量的大小。
    BLOCK_SIZE: tl.constexpr,  # 每个核应该处理的元素数量。
    # 注意：`constexpr` 表示它可以在编译时确定，因此可以作为形状（shape）值使用。
):
    pid = tl.program_id(axis=0)  # 我们使用一维的grid，因此轴为0。
    # 当前核将处理的数据在内存中相对于起始地址的偏移。
    # 例如，如果你有一个长度为256的向量，且块大小（block_size）为64，那么各个程序
    # 将分别访问元素 [0:64, 64:128, 128:192, 192:256]。
    # 注意，offsets 是一个指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建一个掩码（mask），以防止内存操作访问越界。
    mask = offsets < n_elements
    # 加载x和y，并使用掩码屏蔽掉多余的元素，以防输入向量的长度不是块大小的整数倍。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回。
    tl.store(output_ptr + offsets, output, mask=mask)
```

说明：
1. Triton-Ascend默认采取benchmark的方式取片上计算时间，当设置环境变量`export TRITON_BENCH_METHOD="npu"`后，会通过`torch_npu.profiler.profile`的方式取每个kernel配置下的片上计算时间，对于一些triton kernel计算非常快的情况，例如小shape算子，相较于默认方式能够获取更准确的计算时间，但是会显著增加整体autotune的时间，请谨慎开启
2. 目前该进阶用法主要针对的是 Vector 类算子，对于 Cube 类算子自动生成的配置或许性能不佳，待后续优化。更多进阶使用示例可以参考[autotune进阶使用示例](https://gitee.com/ascend/triton-ascend/tree/master/ascend/examples/autotune_cases)

## 更多功能
### 自动生成最优配置的 Profiling 结果
```Python
# 自动在`auto_profile_dir`目录中生成当前autotune最优kernel配置的profiling结果，即利用`torch_npu.profiler.profile`采集的性能数据
# 在社区autotune用法和进阶autotune用法中均可生效
@triton.autotune(
    auto_profile_dir="./profile_result",
    ...
)
```
