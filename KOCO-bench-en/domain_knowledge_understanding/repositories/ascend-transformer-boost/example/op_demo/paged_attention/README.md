# 加速库PagedAttentionOperation C++ Demo
## 介绍
该目录下为加速库PagedAttentionOperation C++调用示例。

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
    - 提供的build脚本仅用于编译和运行paged_attention_demo.cpp，如需编译其他demo，需要替换“paged_attention_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/paged_attention/

## 产品支持情况
本op在Atlas A2/A3系列和Atlas 推理系列产品上实现有所区别

## 场景说明
提供demo编译运行时需要对应更改build脚本：  
1. 不开启并行解码且带mask场景：
    - paged_attention_demo.cpp  
        默认编译脚本可编译运行,该demo仅支持在Atlas A2/A3系列上运行。  
        **参数设置**：

        | 成员名称        | 取值                      |
        | :------------- | :------------------------ |
        | headNum        | 32                        |
        | qkScale        | 1 / sqrt(HEAD_SIZE)       |
        | kvHeadNum      | 32                        |
        | batchRunStatus | 0                         |
        | quantType      | `TYPE_QUANT_UNQUANT`      |
        | hasQuantOffset | false                     |
        | calcType       | `CALC_TYPE_UNDEFINED`     |
        | compressType   | `COMPRESS_TYPE_UNDEFINED` |
        | maskType       | `MASK_TYPE_NORM`          |
        | mlaVHeadSize   | 0                         |
        
        **数据规格**：

        | tensor名字    | 数据类型 | 数据格式   | 维度信息            | cpu/npu |
        | ------------- | -------- | -------- | ------------------ |-------- |
        | `query`       | float16  | nd       | [2, 32, 128]       | npu     |
        | `keyCache`    | float16  | nd       | [16, 128, 32, 128] | npu     |
        | `valueCache`  | float16  | nd       | [16, 128, 32, 128] | npu     |
        | `blockTables` | int32    | nd       | [2, 8]             | npu     |
        | `contextLens` | int32    | nd       | [2]                | cpu     |
        | `mask`        | int32    | nd       | [2, 1, 1024]       | npu     |
        | `attnOut`     | float16  | nd       | [2, 32, 128]       | npu     |

   - paged_attention_qwen_demo.cpp  
        该demo仅支持在Atlas A2/A3系列上运行。  
        **参数设置**：

        | 成员名称        | 取值                      |
        | :------------- | :------------------------ |
        | headNum        | 5                         |
        | qkScale        | 1 / sqrt(HEAD_SIZE)       |
        | kvHeadNum      | 1                         |
        | batchRunStatus | 0                         |
        | quantType      | `TYPE_QUANT_UNDEFINED`    |
        | hasQuantOffset | false                     |
        | calcType       | `CALC_TYPE_UNDEFINED`     |
        | compressType   | `COMPRESS_TYPE_UNDEFINED` |
        | maskType       | `UNDEFINED`               |
        | mlaVHeadSize   | 0                         |

        **数据规格**：

        | tensor名字    | 数据类型  | 数据格式 |  维度信息         | cpu/npu |
        | ------------- | -------- | -------- | ---------------- |---------|
        | `query`       | bf16     | nd       | [1, 5, 128]      | npu     |
        | `qkScale`     | bf16     | nd       | [9, 128, 1, 128] | npu     |
        | `valueCache`  | bf16     | nd       | [9, 128, 1, 128] | npu     |
        | `blockTables` | int32    | nd       | [1, 8]           | npu     |
        | `contextLens` | int32    | nd       | [1]              | cpu     |
        | `attnOut`     | bf16     | nd       | [1, 5, 128]      | npu     |

2. 不带mask：
   - paged_attention_inference_demo.cpp  
    该demo仅支持在Atlas推理系列产品上运行。  
    **参数设置**：

        | 成员名称        | 取值                      |
        | :------------- | :------------------------ |
        | headNum        | 32                        |
        | qkScale        | 1 / sqrt(HEAD_SIZE)       |
        | kvHeadNum      | 32                        |
        | batchRunStatus | 0                         |
        | quantType      | `TYPE_QUANT_UNQUANT`      |
        | hasQuantOffset | false                     |
        | calcType       | `CALC_TYPE_UNDEFINED`     |
        | compressType   | `COMPRESS_TYPE_UNDEFINED` |
        | maskType       | `UNDEFINED`               |
        | mlaVHeadSize   | 0                         |

        **数据规格**：

        | tensor名字    | 数据类型  | 数据格式  | 维度信息            | cpu/npu |
        | ------------- | -------- | -------- | ------------------- |---------|
        | `query`       | bf16     | nd       | [2, 32, 128]        | npu     |
        | `qkScale`     | bf16     | nd       | [16, 1024, 128, 16] | npu     |
        | `valueCache`  | bf16     | nd       | [16, 1024, 128, 16] | npu     |
        | `blockTables` | int32    | nd       | [2, 8]              | npu     |
        | `contextLens` | int32    | nd       | [2]                 | cpu     |
        | `attnOut`     | bf16     | nd       | [2, 32, 128]        | npu     |