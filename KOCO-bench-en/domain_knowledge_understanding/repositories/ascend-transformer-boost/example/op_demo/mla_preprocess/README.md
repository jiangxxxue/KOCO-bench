# 加速库MlaPreprocessOperation C++ Demo
## 介绍
该目录下为加速库MlaPreprocessOperation C++调用示例。

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
    - 提供的build脚本仅用于编译和运行mlapo_demo.cpp，如需编译其他demo，需要替换“mlapo_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/mla_preprocess/

### 场景说明
提供demo分别对应不同产品/场景，具体场景区别可以参见官网。
用户调用demo二进制文件时key输入`dtype`，`tokenNum`，`headNum`，控制数据类型和shape。
    `dtype`对应input的数据类型，分别支持float16/bf16，对应下面数据规格的俩列。

mlapo_demo.cpp：
+ 场景：MLAPO int8量化叠加rope切分
  + ctkv和qNope经过per_head静态对称量化为int8类型；并使用rope拆分kvcache和query，并且krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型。

**参数设置**：
| 成员名称  | 取值         |
| :-------- | :----------- |
| cacheMode | INT8_NZCACHE |

**数据规格**：
| tensor名字     | 数据类型     | 数据格式 | 维度信息                        |
| -------------- | ------------ | -------- | ------------------------------- |
| `input`        | float16/bf16 | nd       | [tokenNum, 7168]                |
| `gamma0`       | float16/bf16 | nd       | [7168]                          |
| `beta0`        | float16/bf16 | nd       | [7168]                          |
| `quantScale0`  | float16/bf16 | nd       | [1]                             |
| `quantOffset0` | int8         | nd       | [1]                             |
| `wdqkv`        | int8         | nz       | [1, 224, 2112, 32]              |
| `deScale`      | int64/float  | nd       | [2112]                          |
| `bias0`        | int32        | nd       | [2112]                          |
| `gamma1`       | float16/bf16 | nd       | [1536]                          |
| `beta1`        | float16/bf16 | nd       | [1536]                          |
| `quantScale1`  | float16/bf16 | nd       | [1]                             |
| `quantOffset1` | int8         | nd       | [1]                             |
| `wuq`          | int8         | nz       | [1, 48, 24576, 32]              |
| `deScale1`     | int64/float  | nd       | [24576]                         |
| `bias1`        | int32        | nd       | [headNum * 192]                 |
| `gamma2`       | float16/bf16 | nd       | [512]                           |
| `cos`          | float16/bf16 | nd       | [tokenNum, 64]                  |
| `sin`          | float16/bf16 | nd       | [tokenNum, 64]                  |
| `wuk`          | float16/bf16 | nz       | [headNum, 32, 128, 16]          |
| `kvCache`      | int8         | nz       | [64, headNum * 512/32, 128, 32] |
| `kvCacheRope`  | float16/bf16 | nd       | [64, headNum * 64/16, 128, 16]  |
| `slotmapping`  | int32        | nd       | [tokenNum]                      |
| `ctkvScale`    | float16/bf16 | nd       | [1]                             |
| `qNopeScale`   | float16/bf16 | nd       | [headNum]                       |
| **Output**     |
| `qOut0`        | int8         | nd       | [tokenNum, headNum, 512]        |
| `kvCacheOut0`  | int8         | nz       | [64, headNum * 512/32, 128, 32] |
| `qOut1`        | float16/bf16 | nd       | [tokenNum, headNum, 64]         |
| `kvCacheOut1`  | float16/bf16 | nz       | [64, headNum * 64/16, 128, 16]  |
> 默认值：`dtype = float16`, `tokenNum = 4`, `headNum = 128`


mlapo_ds_demo.cpp：
+ 场景：MLAPO deepSeek场景
  + 使用rope拆分kvcache和query，per_tensor静态非对称量化。

**参数设置**：
| 成员名称     | 取值         |
| :----------- | :----------- |
| wdqDim       | 1536         |
| qRopeDim     | 64           |
| kRopeDim     | 64           |
| epsilon      | 1e-5         |
| qRotaryCoeff | 2            |
| kRotaryCoeff | 2            |
| transposeWdq | true         |
| transposeWuq | true         |
| transposeWuk | true         |
| cacheMode    | INT8_NZCACHE |

**数据规格**：
| tensor名字     | 数据类型     | 数据格式 | 维度信息                 |
| -------------- | ------------ | -------- | ------------------------ |
| `input`        | float16/bf16 | nd       | [tokenNum, 7168]         |
| `gamma0`       | float16/bf16 | nd       | [7168]                   |
| `beta0`        | float16/bf16 | nd       | [7168]                   |
| `quantScale0`  | float16/bf16 | nd       | [1]                      |
| `quantOffset0` | int8         | nd       | [1]                      |
| `wdqkv`        | int8         | nz       | [2112, 7168]             |
| `deScale`      | int64/float  | nd       | [2112]                   |
| `bias0`        | int32        | nd       | [2112]                   |
| `gamma1`       | float16/bf16 | nd       | [1536]                   |
| `beta1`        | float16/bf16 | nd       | [1536]                   |
| `quantScale1`  | float16/bf16 | nd       | [1]                      |
| `quantOffset1` | int8         | nd       | [1]                      |
| `wuq`          | int8         | nz       | [24576, 1536]            |
| `deScale1`     | int64/float  | nd       | [24576]                  |
| `bias1`        | int32        | nd       | [headNum * 192]          |
| `gamma2`       | float16/bf16 | nd       | [512]                    |
| `cos`          | float16/bf16 | nd       | [tokenNum, 64]           |
| `sin`          | float16/bf16 | nd       | [tokenNum, 64]           |
| `wuk`          | float16/bf16 | nd       | [tokenNum, 128, 512]     |
| `kvCache`      | float16/bf16 | nd       | [161, 128, 1, 512]       |
| `kvCacheRope`  | float16/bf16 | nd       | [161, 128, 1, 64]        |
| `slotmapping`  | int32        | nd       | [tokenNum]               |
| `ctkvScale`    | float16/bf16 | nd       | [1]                      |
| `qNopeScale`   | float16/bf16 | nd       | [1]                      |
| **Output**     |
| `qOut0`        | int8         | nd       | [tokenNum, headNum, 512] |
| `kvCacheOut0`  | int8         | nz       | [161, 128, 1, 512]       |
| `qOut1`        | float16/bf16 | nd       | [tokenNum, headNum, 64]  |
| `kvCacheOut1`  | float16/bf16 | nz       | [161, 128, 1, 564]       |
> 默认值：`dtype = float16`, `tokenNum = 32`, `headNum = 128`
