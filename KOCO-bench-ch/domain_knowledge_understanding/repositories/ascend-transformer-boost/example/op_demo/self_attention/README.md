# 加速库SelfAttentionOperation C++ Demo
## 介绍
该目录下为加速库SelfAttentionOperation C++调用示例。

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
    - 提供的build脚本仅用于编译和运行self_attention_encoder_demo.cpp，如需编译其他demo，需要替换“self_attention_encoder_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/self_attention/

## 产品支持情况
SelfAttention在Atlas A2/A3系列仅支持部分场景，且Encoder场景在Atlas 推理系列产品上调用与Atlas A2/A3有所区别

### 场景说明
提供demo分别对应不同产品/场景，具体场景区别可以参见官网，编译运行时需要对应更改build脚本：

#### self_attention_encoder_demo.cpp
+ 场景：FA Encoder基础场景，分开传入key，CacheK，value和CacheV
+ 默认编译脚本可编译运行
+ 该demo仅支持在Atlas A2/A3上运行
+ demo中使用全量的上三角mask演示

**参数设置**：
| 成员名称   | 取值                 |
| :--------- | :------------------- |
| headNum    | 32                   |
| kvHeadNum  | 32                   |
| qkScale    | 1/sqrt(128)          |
| calcType   | `ENCODER`            |
| kernelType | `KERNELTYPE_DEFAULT` |
| maskType   | `MASK_TYPE_NORM`     |

**数据规格**：
| tensor名字      | 数据类型 | 数据格式 | 维度信息            | cpu/npu |
| --------------- | -------- | -------- | ------------------- | ------- |
| `query`         | float16  | nd       | [160,2048]          | npu     |
| `key`           | float16  | nd       | [160,2048]          | npu     |
| `value`         | float16  | nd       | [160,2048]          | npu     |
| `cacheK`        | float16  | nd       | [1, 10, 1024, 2048] | npu     |
| `cacheV`        | float16  | nd       | [1, 10, 1024, 2048] | npu     |
| `attentionMask` | float16  | nd       | [10, 1024, 1024]    | npu     |
| `tokenOffset`   | int32    | nd       | [10]                | cpu     |
| `seqLen`        | int32    | nd       | [10]                | cpu     |
| `layerId`       | int32    | nd       | [1]                 | npu     |
| **Output**      |
| `output`        | float16  | nd       | [1]                 | npu     |
+ q，k，v第一维度为总词元长度，即`sum(seqlen)`
+ q，k，v第二维度headNum，headSize合轴，实际为headHum(32) $\times$ headSize(128)

#### self_attention_encoder_inference_demo.cpp
+ 场景：fa encoder基础场景在Atlas推理系列上的实现，分开传入key，CacheK，value和CacheV
+ 更改编译脚本为：
 `g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" self_attention_encoder_inference_demo.cpp demo_util.h -l atb -l ascendcl -o self_attention_encoder_inference_demo`
- 运行时调用：
`./self_attention_encoder_inference_demo`
+ 该demo仅支持在Atlas推理系列上运行

**参数设置**：
| 成员名称   | 取值                  |
| :--------- | :-------------------- |
| headNum    | 16                    |
| kvHeadNum  | 16                    |
| qkScale    | 1/sqrt(128)           |
| calcType   | `ENCODER`             |
| kernelType | `KERNELTYPE_DEFAULT`  |
| maskType   | `MASK_TYPE_UNDEFINED` |

**数据规格**：
| tensor名字    | 数据类型 | 数据格式 | 维度信息            | cpu/npu |
| ------------- | -------- | -------- | ------------------- | ------- |
| `query`       | float16  | nd       | [16, 256]           | npu     |
| `key`         | float16  | nd       | [16, 256]           | npu     |
| `value`       | float16  | nd       | [16, 256]           | npu     |
| `cacheK`      | float16  | nd       | [1, 1, 16, 256, 16] | npu     |
| `cacheV`      | float16  | nd       | [1, 1, 16, 256, 16] | npu     |
| `tokenOffset` | int32    | nd       | [1]                 | cpu     |
| `seqLen`      | int32    | nd       | [1]                 | cpu     |
| `layerId`     | int32    | nd       | [1]                 | npu     |
| **Output**    |
| `output`      | float16  | nd       | [16, 256]           | npu     |

#### self_attention_pa_encoder_demo.cpp
+ 场景：FA使用PA Encoder的场景，使用FA输入，只需传入key，value
  + 传入不同的headNum，kvHeadNum且headNum可以被kvHeadNum时开启GQA（Grouped Query Attention）。
+ 该demo仅支持在Atlas A2/A3系列上运行
+ demo中使用全量的上三角mask演示

**参数设置**：
| 成员名称   | 取值                 |
| :--------- | :------------------- |
| headNum    | 32                   |
| kvHeadNum  | 16                   |
| qkScale    | 1/sqrt(128)          |
| calcType   | `PA_ENCODER`         |
| kernelType | `KERNELTYPE_DEFAULT` |
| maskType   | `MASK_TYPE_NORM`     |

**数据规格**：
| tensor名字      | 数据类型 | 数据格式 | 维度信息        | cpu/npu |
| --------------- | -------- | -------- | --------------- | ------- |
| `query`         | float16  | nd       | [3072, 32, 128] | npu     |
| `key`           | float16  | nd       | [3072, 16, 128] | npu     |
| `value`         | float16  | nd       | [3072, 16, 128] | npu     |
| `attentionMask` | float16  | nd       | [4, 1024, 1024] | npu     |
| `seqLen`        | int32    | nd       | [4]             | cpu     |
| **Output**      |
| `output`        | float16  | nd       | [3072, 32, 128] | npu     |

####  self_attention_pa_encoder_qwen_demo.cpp
+ 场景：FA使用PA Encoder的场景，使用FA输入，只需传入key，value
  + 传入不同的headNum，kvHeadNum且headNum可以被kvHeadNum时开启GQA（Grouped Query Attention）。
+ 该demo仅支持在Atlas A2/A3系列上运行
+ demo中为了应对长序列的情况，使用压缩上三角mask演示

**参数设置**：
| 成员名称   | 取值                      |
| :--------- | :------------------------ |
| headNum    | 5                         |
| kvHeadNum  | 1                         |
| qkScale    | 1/sqrt(128)               |
| isTriuMask | 1                         |
| calcType   | `PA_ENCODER`              |
| kernelType | `KERNELTYPE_DEFAULT`      |
| maskType   | `MASK_TYPE_NORM_COMPRESS` |

**数据规格**：
| tensor名字      | 数据类型 | 数据格式 | 维度信息       | cpu/npu |
| --------------- | -------- | -------- | -------------- | ------- |
| `query`         | float16  | nd       | [1024, 5, 128] | npu     |
| `key`           | float16  | nd       | [1024, 1, 128] | npu     |
| `value`         | float16  | nd       | [1024, 1, 128] | npu     |
| `attentionMask` | float16  | nd       | [128, 128]     | npu     |
| `seqLen`        | int32    | nd       | [1]            | cpu     |
| **Output**      |
| `output`        | float16  | nd       | [1024, 5, 128] | npu     |

#### self_attention_prefix_encoder_demo.cpp
+ 场景：FA使用Prefix Encoder的场景，传入PA的依据blockTables存放的key，value
  + 此场景支持q和kv不等长，但是要求：
    $$\forall i \lt len(seqLen)，kvSeqLen[i] - seqLen[i] = 0 \ (mod \ 128) $$
+ 该demo仅支持在Atlas A2/A3系列上运行
+ demo中使用Alibi上三角mask叠加bias slopes演示

**参数设置**：
| 成员名称   | 取值                        |
| :--------- | :-------------------------- |
| headNum    | 32                          |
| kvHeadNum  | 8                           |
| qkScale    | 1/sqrt(128)                 |
| isTriuMask | 1                           |
| calcType   | `PREFIX_ENCODER`            |
| kernelType | `KERNELTYPE_HIGH_PRECISION` |
| maskType   | `MASK_TYPE_ALIBI_COMPRESS`  |


**数据规格**：
| tensor名字    | 数据类型 | 数据格式 | 维度信息           | cpu/npu |
| ------------- | -------- | -------- | ------------------ | ------- |
| `query`       | float16  | nd       | [96, 32, 128]      | npu     |
| `key`         | float16  | nd       | [480, 128, 8, 128] | npu     |
| `value`       | float16  | nd       | [480, 128, 8, 128] | npu     |
| `blockTables` | float16  | nd       | [4, 4]             | npu     |
| `mask`        | float16  | nd       | [32, 96, 128]      | npu     |
| `seqLen`      | int32    | nd       | [4]                | cpu     |
| `kvSeqLen`    | int32    | nd       | [4]                | cpu     |
| `slopes`      | float32  | nd       | [128]              | npu     |
| **Output**    |
| `output`      | float16  | nd       | [96, 32, 128]      | npu     |
