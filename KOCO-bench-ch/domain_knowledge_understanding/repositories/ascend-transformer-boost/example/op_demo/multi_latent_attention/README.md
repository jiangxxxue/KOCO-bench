# 加速库MultiLatentAttentionOperation C++ Demo
## 介绍
该目录下为加速库MultiLatentAttentionOperation C++调用示例。

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
tests/apitest/opstest/python/operations/multi_latent_attention/

## 场景说明

  该算子所给demo仅支持在Atlas A2/A3系列产品上运行，demo的场景说明如下：

- mlapa_demo.cpp
  
    **参数设置**：
    | 成员名称   | 取值                  |
    | :-------- | :-------------------- |
    | headNum   | 128                   |
    | qkScale   | 1/sqrt(576)*          |
    | kvHeadNum | 1                     |
    | maskType  | `UNDEFINED`           |
    | calcType  | `CALC_TYPE_UNDEFINED` |
    | cacheMode | `INT8_NZCACHE`        |

    > **注意**：qkScale设置值为MLA做rope投影前的headSize，即`512(原始) + 64(投影) = 576`

    **数据规格**：
    | tensor名字    | 数据类型  | 数据格式  | 维度信息          | cpu/npu |
    | ------------- | -------- | -------- | ----------------- | ------- |
    | `qNope`       | int8     | nd       | [4, 128, 512]     | npu     |
    | `qRope`       | float16  | nd       | [4, 128, 64]      | npu     |
    | `ctKV`        | int8     | nz       | [48, 16, 128, 32] | npu     |
    | `kRope`       | float16  | nz       | [48, 4, 128, 16]  | npu     |
    | `blockTables` | int32    | nd       | [4, 12]           | npu     |
    | `contextLens` | int32    | nd       | [4]               | cpu     |
    | `qkDescale`   | float    | nd       | [128]             | npu     |
    | `pvDescale`   | float    | nd       | [128]             | npu     |
    | `attenOut`    | float16  | nd       | [4, 128, 512]     | npu     |

- mlapa_ds_demo.cpp  

    **参数设置**：
    | 成员名称   | 取值                  |
    | :-------- | :-------------------- |
    | headNum   | 128                   |
    | qkScale   | 0.1352667747812271    |
    | kvHeadNum | 1                     |
    | maskType  | `UNDEFINED`           |
    | calcType  | `CALC_TYPE_UNDEFINED` |
    | cacheMode | `KROPE_CTKV`          |

    **数据规格**：
    | tensor名字     | 数据类型  | 数据格式  | 维度信息           | cpu/npu |
    | ------------- | -------- | -------- | ------------------ |-------- |
    | `qNope`       | float16  | nd       | [32, 128, 512]     | npu     |
    | `qRope`       | float16  | nd       | [7168, 128, 64]    | npu     |
    | `ctKV`        | float16  | nd       | [160, 128, 1, 512] | npu     |
    | `kRope`       | float16  | nd       | [160, 128, 1, 64]  | npu     |
    | `blockTables` | int32    | nd       | [32, 5]            | npu     |
    | `contextLens` | int32    | nd       | [32]               | cpu     |
    | `attenOut`    | float16  | nd       | [32, 128, 512]     | npu     |
