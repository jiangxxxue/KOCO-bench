# 加速库RingMla C++ Demo
## 介绍
该目录下为加速库RingMla C++调用示例。

## 使用说明
- 首先source 对应的CANN和nnal包
    1. source [cann安装路径]/set_env.sh
        默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. source [nnal安装路径]/set_env.sh
        默认：source /usr/local/Ascend/nnal/atb/set_env.sh
        1. 如果使用加速库源码编译，source [加速库源码路径]/output/atb/set_env.sh
        例如： source ./ascend-transformer-boost/output/atb/set_env.sh

- 运行demo
    ```sh
    bash build.sh
    ```
    **注意**：
    - 使用cxx_abi=0（默认）时，设置`D_GLIBCXX_USE_CXX11_ABI`为0，即：
        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=0 -I ...
        ```
    - 使用cxx_abi=1时，更改`D_GLIBCXX_USE_CXX11_ABI`为1，即：
        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=1 -I ...
        ```

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/ring_mla/

## 产品支持情况
RingMla仅Atlas A2/A3系列

### 场景说明
1. RingMla：
    + 基础场景，对于query，key分别传入带与不带rope转置的矩阵
    + 传入固定shape，512x512的上三角mask
    + 默认编译脚本可编译运行
    + 该demo仅支持在Atlas A2/A3系列上运行

#### Demo分段参数/tensor规格设置
因为ringMLA和其他的算子较为不同，第一次运算时不带前次生成的prevOut和prevLse，但是从第二次起使用时需要带上，这里分俩段描述：
1. 第一轮

**参数设置**：
| 成员名称   | 取值                 |
| :--------- | :------------------- |
| calcType   | CALC_TYPE_FISRT_RING |
| headNum    | 16                   |
| kvHeadNum  | 8                    |
| qkScale    | 1/sqrt(192)          |
| kernelType | `KERNELTYPE_DEFAULT` |
| maskType   | `MASK_TYPE_TRIU`     |
> **注意**：qkScale设置值为RingMLA做rope转置前query，key合一的headSize，即`128(nope) + 64(rope) = 192`

**数据规格**：
| tensor名字   | 数据类型 | 数据格式 | 维度信息        | cpu/npu |
| ------------ | -------- | -------- | --------------- | ------- |
| `queryNope`  | bf16     | nd       | [1228, 16, 128] | npu     |
| `queryRope`  | bf16     | nd       | [1228, 16, 64]  | npu     |
| `keyNope`    | bf16     | nd       | [828, 8, 128]   | npu     |
| `keyRope`    | bf16     | nd       | [828, 8, 64]    | npu     |
| `value`      | bf16     | nd       | [828, 8, 128]   | npu     |
| `mask`       | bf16     | nd       | [512, 512]      | npu     |
| `seqLen`     | bf16     | nd       | [2, 3]          | cpu     |
| **Output**   |
| `output`     | bf16     | nd       | [1228, 16, 128] | npu     |
| `softmaxLse` | float    | nd       | [16, 1228]      | npu     |
> q第一维度为总词元长度，对应`sum(seqlen[0])`，k，v第一维度对应`sum(seqlen[1])`

1. 第二轮
第二轮中会使用第一轮新生成的output和softmaxLse进行计算。

**参数设置**：
| 成员名称     | 取值                 |
| :----------- | :------------------- |
| **calcType** | CALC_TYPE_DEFAULT    |
| headNum      | 16                   |
| kvHeadNum    | 8                    |
| qkScale      | 1/sqrt(192)          |
| kernelType   | `KERNELTYPE_DEFAULT` |
| maskType     | `MASK_TYPE_TRIU`     |
> 需要额外更改param里的calcType为CALC_TYPE_DEFAULT，其他保持一致

**数据规格**：
| tensor名字    | 数据类型 | 数据格式 | 维度信息        | cpu/npu |
| ------------- | -------- | -------- | --------------- | ------- |
| `queryNope`   | bf16     | nd       | [1228, 16, 128] | npu     |
| `queryRope`   | bf16     | nd       | [1228, 16, 64]  | npu     |
| `keyNope`     | bf16     | nd       | [828, 8, 128]   | npu     |
| `keyRope`     | bf16     | nd       | [828, 8, 64]    | npu     |
| `value`       | bf16     | nd       | [828, 8, 128]   | npu     |
| `mask`        | bf16     | nd       | [512, 512]      | npu     |
| `seqLen`      | bf16     | nd       | [2, 3]          | cpu     |
| **`prevOut`** | bf16     | nd       | [1228, 16, 128] | npu     |
| **`prevLse`** | float    | nd       | [16, 1228]      | npu     |
| **Output**    |
| `output`      | bf16     | nd       | [1228, 16, 128] | npu     |
| `softmaxLse`  | float    | nd       | [16, 1228]      | npu     |
> 第二轮使用第一轮新生成的output和softmaxLse来作为prevOut和prevLse。
