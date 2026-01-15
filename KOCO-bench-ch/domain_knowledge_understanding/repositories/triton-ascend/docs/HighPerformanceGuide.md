# 高性能Triton算子编程开发指南
### 1. 合理设置并发任务个数与Tile/Block切块大小
开发者在编写Triton算子时，一个重要的设计工作就是确定并发任务个数以及Tile/Block切块大小，而这两者间通常是相互关联的。针对给定的计算任务/输入输出数据形状规格，我们可以先确定Tile/Block切块的大小，然后推算出需要并发的任务个数，反之亦然。例如，对于Gelu算子，该算子的输入Tensor *A* 与输出Tensor *B* 的形状为[M, N], 假设我们选定的Tile/Block切块大小为[m, n], 那么并发任务个数即为 (M/m)*(N/n) 。
#### 1.1 昇腾平台上设定并发任务个数的推荐方案
昇腾 NPU 平台具备多个计算核心（即aicore，包括cube/vector两类），具体个数与底层芯片型号相关，底层物理aicore个数可以通过driver.active.utils.get_device_properties接口获取。虽然运行时接口允许在执行Triton kernel时启动多于底层物理aicore个数的并发任务（最大并发任务个数不得超过65535），但当并发任务数多于底层物理核数时，这些并发任务实际将划分为多个批次调度到NPU上运行，单个批次内的并行任务个数依然不能超过底层物理aicore个数。分批调度会引入额外的设备侧开销，从而影响Triton算子整体执行性能。

**为能最大化利用NPU的物理aicore资源进行并行计算加速，同时避免分批调度开销，建议开发者将并发任务个数配置为底层aicore个数。对于仅涉及Vector计算的Triton算子，并发任务个数应等于vector core的个数；其他类型的Triton算子（即Triton算子内使用了tl.dot），并发任务个数应等于aicore的个数。**

#### 1.2 昇腾平台上设定Tile/Block切块大小的推荐方案
确定了并行任务个数后，每个aicore上实际处理的单个任务所对应的任务求解规模就随之确定下来，即单个任务对应的Tile/Block切块所包含的元素个数等于总问题规模除以并行任务个数。同样以Gelu算子为例，输入Tensor *A*的形状为[M, N]，那么总问题规模为 M*N ，当并行任务个数为*P* 时，Tile/Block切块大小为 M*N/P 。
然而，由于NPU上单aicore的片上内存空间有限，按照上述方法得到的Tile/Block切块可能无法完整搬入片上内存，导致编译时出现片上内存无法分配的报错。此外，由于编译器会尝试为Triton算子的中间计算结果分配片上存储空间，或是为使能并行流水而开启多级缓存，因此如何确定Triton算子每次load/store的数据量就显得至关重要，对Triton算子的功能与性能均有影响。

**我们推荐采用以下多层次的Tile/Block切块大小设定方案**，常见的切分参数包括：
```
block_size：单个aicore上处理的Tile/Block切块大小（核间数据切分）
sub_block_size: 单个aicore上单次运算的Sub-Tile/Sub-Block切块大小（核内数据切分）
```
其中，sub_block_size的上限大小可以根据NPU单aicore可使用的片上存储空间（即cube核L1缓存或vector核UB缓存）的大小计算得出。例如，Atlas 800T/I A2产品的片上内存容量为192KB，因此单次运算所涉及的输入、输出、中间计算结果的总大小不得超过该上限。**开发者可以利用Triton的Autotune能力，选择使能最优性能的sub_block_size。**

#### 1.3 Gelu算子示例代码
```
import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch
import torch_npu

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

def gelu(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res

@triton.jit
def triton_gelu(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    #计算当前核处理数据块的起始偏移地址，实现核间切分。每个核仅负责 XBLOCK 大小的数据范围。
    xoffset = tl.program_id(0) * XBLOCK
	#在单个核内部进一步细分数据块，每次处理 XBLOCK_SUB 大小的数据，实现核内切分。
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
	    #构造当前迭代的数据索引数组，用于访问输入和输出张量。
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
		#设置掩码以防止越界访问，确保只处理合法范围内的数据。
        xmask = x_index < xnumel
		#从全局存储空间将输入数据加载到片上内存
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
		#从片上内存将输出数据加载到全局存储空间
        tl.store(out_ptr0 + x_index, ret, xmask)

def test_gelu(shape, NUMEL):
    print(f"input : shape = {shape} NUMEL = {NUMEL}")
    in = torch.rand(size=shape, dtype=torch.float32).npu()

    ans = gelu(in)
    print(f"gelu output: ans = {ans}")

    out = torch.zeros(size=shape, dtype=torch.float32).npu()
	#gelu仅涉及vector运算，因此根据vector核数确定block_size
	#对于非纯vector计算类的算子，应使用get_npu_properties()["num_aicore"]获得物理核数总数
    num_core = get_npu_properties()["num_vectorcore"]
    block_size = in.numel()/ncore
	#片上内存大小为192KB，对于float32类型数据，则理论可容纳的最大数据量为192*1024/4=49152；
	#由于需为中间变量等预留片上空间，因此，此示例中选择sub_block_size=49152*0.25=8192
	#建议使用autotune对sub_block_size参数进行寻优已获得更好的性能
    sub_block_size = 8192
    triton_gelu[num_core, 1, 1](in, out, in.numel(), block_size, sub_block_size)
    print(f"triton_gelu output: out = {out}")

    torch.allclose(out, ans,  rtol=1e-03, atol=1e-03, equal_nan=True)
    print(f"test Pass")

test_gelu((32, 32768), 32*32768)
```

### 2. 合理设计Tile/Block数据读写方式与顺序
文档更新中...


### 3. 合理使用针对昇腾平台新增的Triton Python API
除了Triton社区提供的Python API之外，我们新增了少量拓展API。利用这些API，开发者能够控制Triton Ascend编译器实施代码优化，从而结合开发者的经验获得更好的二进制代码性能。我们仍在不断完善与更新这些拓展API，其接口设计可能会在未来发生变化，开发者请酌情使用。

拓展API的功能说明如下：

 **triton.language.multibuffer(tensor, buffer_num)** : 控制编译器针对tensor数据开启buffer_num个缓冲存储区，使能数据搬运与数据计算并行；使用该API能够提升性能，但同时内存使用量为未使用时的buffer_num倍。[参考样例代码](./ascend/examples/pytest_ut/test_compile_hint.py) 


### 4. 查阅编译过程产生的临时文件

Triton Ascend的编译过程中将产生一系列编译产物，包括IR文件以及object文件，开发者可以通过分析IR，判断编译结果是否满足预期。

- #### 4.1 临时文件路径
~/.triton/cache目录是默认的cache缓存位置，~/.triton/dump/目录是默认的临时文件存储位置。这两个路径下主要缓存了编译过程中生成的中间产物和最终结果（包括.ttadapter、.ttir、.so后缀的文件），以提高编译与执行效率，同时方便开发者分析与调试。
```
fn_npu_.ttadapter # 将 Triton 写的高级代码转换为适合目标硬件（Ascend NPU）的形式。
fn_npu_.ttir      # 从 ttadapter 文件生成 IR 文件，源代码已经被转化为一种更接近机器码的形式。
launcher_cxx11abi1.cpython-311-aarch64-linux-gnu.so  # 编译生成的共享库文件，文件名表示了python311版本，CPU架构，打包成可以被python执行的动态链接库。
kernel.ttadapter.mlir   # 将Triton高级语言描述的内核转化为MLIR格式
kernel.ttir.mlir        # 是MLIR格式的文件，但这个文件代表的是更接近目标硬件的中间表示。
launcher_cxx11abi1.cxx  # 用于封装和调用前面步骤中生成的内核代码，使其能够在Python环境中被调用
```
- #### 4.2 缓存使能方法
```
# 设置环境变量DEBUG
export TRITON_DEBUG=1

# 运行某Triton用例前，先清理~/.triton/dump和~/.triton/cache下的文件，再执行某算子
python triton_xxx_test.py

# 在窗口打印日志中，找到dump路径，切换到该路径下即可查看相关IR文件。
cd ~/.triton/dump/

```
注： 若设置环境变量TRITON_DEBUG=1，执行Triton算子后，未打印dump路径，可先手动删除临时文件: