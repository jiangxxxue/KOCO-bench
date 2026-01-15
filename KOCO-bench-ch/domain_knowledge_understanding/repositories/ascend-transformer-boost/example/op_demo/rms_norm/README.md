# 加速库RmsNormOperation C++ Demo
## 介绍
该目录下为加速库RmsNormOperation C++调用示例。

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
    - 提供的build脚本仅用于编译和运行rms_norm_demo.cpp，如需编译其他demo，需要替换“rms_norm_demo”为对应的cpp文件名


## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/rms_norm/

### 场景说明
提供demo分别对应，编译运行时需要对应更改build脚本：
- **rms_norm_demo.cpp**

    【注】：默认编译脚本可编译运行

    **参数设置**
    |        Param        |                        value                         |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-5                         |

    **输入**
    | TensorName | DataType | DataFormat |      Shape      |
    | :--------: | :------: | :--------: | :-------------: |
    |     x      | float16  |     nd     | [4, 1024, 5120] |
    |   gamma    | float16  |     nd     |     [5120]      |

    **输出**
    | TensorName | DataType | DataFormat |      Shape      |
    | :--------: | :------: | :--------: | :-------------: |
    |   output   | float16  |     nd     | [4, 1024, 5120] |

    ---

- **rms_norm_qwen_demo_0.cpp**

    【注】：编译脚本内替换 rms_norm_demo.cpp 为 rms_norm_qwen_demo_0.cpp 可编译运行；
           该示例仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。
    **参数设置**
    |        Param        |                        value                         |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **输入**
    | TensorName | DataType | DataFormat |    Shape     |
    | :--------: | :------: | :--------: | :----------: |
    |     x      |   bf16   |     nd     | [1024, 5120] |
    |   gamma    |   bf16   |     nd     |    [5120]    |

    **输出**
    | TensorName | DataType | DataFormat |    Shape     |
    | :--------: | :------: | :--------: | :----------: |
    |   output   |   bf16   |     nd     | [1024, 5120] |

    ---

- **rms_norm_qwen_demo_1.cpp**

    【注】：编译脚本内替换 rms_norm_demo.cpp 为 rms_norm_qwen_demo_1.cpp 可编译运行；
           该示例仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。
    **参数设置**
    |        Param        |                        value                         |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **输入**
    | TensorName | DataType | DataFormat |   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |     x      |   bf16   |     nd     | [1, 5120] |
    |   gamma    |   bf16   |     nd     |  [5120]   |

    **输出**
    | TensorName | DataType | DataFormat |   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |   output   |   bf16   |     nd     | [1, 5120] |

    ---

- **rms_norm_qwen_demo_2.cpp**

    【注】：编译脚本内替换 rms_norm_demo.cpp 为 rms_norm_qwen_demo_2.cpp 可编译运行；
           该示例仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。
    **参数设置**
    |        Param        |                        value                         |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **输入**
    | TensorName | DataType | DataFormat |   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |     x      |   bf16   |     nd     | [5, 5120] |
    |   gamma    |   bf16   |     nd     |  [5120]   |

    **输出**
    | TensorName | DataType | DataFormat |   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |   output   |   bf16   |     nd     | [5, 5120] |

    ---

- **rms_norm_deepseek_demo_0.cpp**

    【注】：编译脚本内替换 rms_norm_demo.cpp 为 rms_norm_deepseek_demo_0.cpp 可编译运行；
           该示例仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。
    **参数设置**
    |        Param        |                        value                         |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **输入**
    | TensorName | DataType | DataFormat |    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |     x      | float16  |     nd     | [512, 7168] |
    |   gamma    | float16  |     nd     |   [7168]    |

    **输出**
    | TensorName | DataType | DataFormat |    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |   output   | float16  |     nd     | [512, 7168] |

    ---

- **rms_norm_deepseek_demo_1.cpp**

    【注】：编译脚本内替换 rms_norm_demo.cpp 为 rms_norm_deepseek_demo_1.cpp 可编译运行；
           该示例仅适用于Atlas A2/A3训练系列产品、Atlas 800I A2推理产品、Atlas A3 推理系列产品。
    **参数设置**
    |        Param        |                        value                         |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **输入**
    | TensorName | DataType | DataFormat |   Shape    |
    | :--------: | :------: | :--------: | :--------: |
    |     x      | float16  |     nd     | [32, 7168] |
    |   gamma    | float16  |     nd     |   [7168]   |

    **输出**
    | TensorName | DataType | DataFormat |   Shape    |
    | :--------: | :------: | :--------: | :--------: |
    |   output   | float16  |     nd     | [32, 7168] |