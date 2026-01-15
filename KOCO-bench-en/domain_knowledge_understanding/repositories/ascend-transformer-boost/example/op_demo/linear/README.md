# 加速库LinearOperation C++ Demo

## 介绍

该目录下为加速库LinearOperation C++调用示例。

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
  - 提供的build脚本仅用于编译和运行linear_demo.cpp，如需编译其他demo，需要替换“linear_demo”为对应的cpp文件名

## 额外说明

示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
`tests/apitest/opstest/python/operations/linear/`

## 产品支持情况

本op在Atlas A2/A3系列和Atlas 推理系列产品上实现有所区别。

### 场景说明

提供demo分别对应，编译运行时需要对应更改build脚本：

1. 基础场景：
    - linear_demo.cpp  
    默认编译脚本可编译运行，若未特别说明，则该demo支持Atlas A2/A3系列和Atlas 推理系列产品上运行。  
        **参数设置**：

        |  成员名称    | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | true               |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **数据规格**:

        | tensor名字 | 数据类型 | 数据格式  | 维度信息  | cpu/npu |
        | :--------- | :------- | :------- | :------- | :------ |
        | `x`        | float16  | nd       | [2, 3]   | npu     |
        | `weight`   | float16  | nd       | [3, 2]   | npu     |
        | `bias`     | float16  | nd       | [1, 2]   | npu     |
        | `output`   | float16  | nd       | [2, 2]   | npu     |

    - linear_ds_demo.cpp  
        该demo仅支持在Atlas A2/A3系列产品上运行。  
        **参数设置**：

        | 成员名称     | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | true               |
        | hasBias     | false              |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **数据规格**：  

        | tensor名字 | 数据类型 | 数据格式  | 维度信息     | cpu/npu |
        | ---------- | -------- | -------- | ----------- | :------ |
        | `x`        | float    | nd       | [512, 7168] | npu     |
        | `weight`   | float    | nd       | [256, 7168] | npu     |
        | `output`   | float    | nd       | [512, 256]  | npu     |

    - linear_qwen_demo.cpp  
        该demo仅支持在Atlas A2/A3系列产品上运行。  
        **参数设置**：

        |  成员名称    | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | false              |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **数据规格**：  

        | tensor名字 | 数据类型  | 数据格式  | 维度信息      | cpu/npu |
        | ---------- | -------- | -------- | ------------ | :------ |
        | `x`        | bf16     | nd       | [1, 1728]    | npu     |
        | `weight`   | bf16     | nz       | [1728, 5120] | npu     |
        | `output`   | bf16     | nd       | [1, 5120]    | npu     |
    - linear_qwen_bias_demo.cpp  
        该demo仅支持在Atlas A2/A3系列产品上运行。  
        **参数设置**：

        | 成员名称     | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | true               |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **数据规格**：  

        | tensor名字 | 数据类型 | 数据格式 | 维度信息       | cpu/npu |
        | ---------- | -------- | -------- | ------------ |  ------ |
        | `x`        | bf16     | nd       | [1024, 5120] | npu     |
        | `weight`   | bf16     | nz       | [5120, 896]  | npu     |
        | `bias`     | bf16     | nd       | [1, 896]     | npu     |
        | `output`   | bf16     | nd       | [1024, 896]  | npu     |

2. 爱因斯坦乘场景：
    linear_einsum_demo.cpp
    - 即更改编译脚本为：
    `g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" linear_einsum_demo.cpp demo_util.h -l atb -l ascendcl -o linear_einsum_demo`
    - 运行时调用：
    `./linear_einsum_demo`
    - 该demo仅支持在Atlas A2/A3系列产品上运行
    - linear_einsum_demo.cpp

        **参数设置**：

        | 成员名称     | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | false              |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_EIN_SUM`   |

        **数据规格**：  

        | tensor名字 | 数据类型  | 数据格式  |  维度信息        | cpu/npu |
        | ---------- | -------- | -------- | --------------- | ------- |
        | `x`        | float16  | nd       | [32, 128, 512]  | npu     |
        | `weight`   | float16  | nd       | [128, 512, 128] | npu     |
        | `output`   | float16  | nd       | [32, 128, 512]  | npu     |
3. 量化场景

    - linear_dequant_demo.cpp  
        该demo仅支持在Atlas A2/A3系列产品上运行。  
        **参数设置**：

        | 成员名称     | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | true               |
        | outDataType | `ACL_BF16`         |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **数据规格**：  

        | tensor名字 | 数据类型  | 数据格式  | 维度信息 | cpu/npu |
        | ---------- | -------- | -------- | -------- |-------- |
        | `x`        | int8     | nd       | [2, 3]   | npu     |
        | `weight`   | int8     | nd       | [3, 2]   | npu     |
        | `bias`     | int32    | nd       | [1, 2]   | npu     |
        | `deqScale` | float    | nd       | [1, 2]   | npu     |
        | `output`   | bf16     | nd       | [2, 2]   | npu     |


    - linear_dequant_ds_demo.cpp  
        该demo支持Atlas A2/A3系列和Atlas 推理系列产品上运行。  
        **参数设置**：

        | 成员名称     | 取值               |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | true               |
        | hasBias     | true               |
        | outDataType | `ACL_FLOAT16`      |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **数据规格**：

        | tensor名字 | 数据类型  | 数据格式  | 维度信息      | cpu/npu |
        | ---------- | -------- | -------- | ------------- |-------- |
        | `x`        | int8     | nd       | [32, 16384]   | npu     |
        | `weight`   | int8     | nd       | [7168, 16384] | npu     |
        | `bias`     | int32    | nd       | [1, 7168]     | npu     |
        | `deqScale` | int64    | nd       | [1, 7168]     | npu     |
        | `output`   | float16  | nd       | [32, 7168]    | npu     |
