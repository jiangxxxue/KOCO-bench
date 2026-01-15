# 加速库PagedCacheLoadOperation C++ Demo
## 介绍
该目录下为加速库PagedCacheLoadOperation C++调用示例，以下示例仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。

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
    - 提供的build脚本仅用于编译和运行paged_cache_load_demo.cpp，如需编译其他demo，需要替换“paged_cache_load_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/paged_cache_load/

### 场景说明
提供demo分别对应，编译运行时需要对应更改build脚本：
- **paged_cache_load_demo.cpp**

    【注】：默认编译脚本可编译运行

    **参数设置**
    |        Param        |                        value                        |
    | :-----------------: | :-------------------------------------------------: |
    |     kvCacheCfg      | atb::infer::PagedCacheLoadParam::K_CACHE_V_CACHE_NZ |
    |    hasSeqStarts     |                        false                        |
    | isSeqLensCumsumMode |                        false                        |

    **输入**
    | TensorName  | DataType | DataFormat |      Shape      |
    | :---------: | :------: | :--------: | :-------------: |
    |  keyCache   |   int8   | fractal_nz | [4, 4, 128, 32] |
    | valueCache  |   int8   | fractal_nz | [4, 4, 128, 32] |
    | blocktable  |  int32   |     nd     |     [3, 1]      |
    | contextlens |  int32   |     nd     |       [3]       |
    |     key     |   int8   |     nd     |   [384, 128]    |
    |    value    |   int8   |     nd     |   [384, 128]    |

    **输出**
    | TensorName | DataType | DataFormat |   Shape    |
    | :--------: | :------: | :--------: | :--------: |
    |    key     |   int8   |     nd     | [384, 128] |
    |   value    |   int8   |     nd     | [384, 128] |