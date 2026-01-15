# 加速库FaupdateOperation C++ Demo
## 介绍
该目录下为加速库FaupdateOperation C++调用示例。

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
    - 提供的build脚本仅用于编译和运行faupdate_demo，如需编译其他demo，需要替换“faupdate_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
`tests/apitest/opstest/python/operations/faupdate/`

## 产品支持情况
本op在Atlas A2/A3系列和Atlas 推理系列产品上实现有所区别

## 场景说明
所给Demo的场景说明如下：  
- faupdate_demo.cpp  
  该demo仅支持在Atlas A2/A3系列产品上运行。  
    **参数设置**

    | 成员名称       | 取值          |
    | :------------- | :------------ |
    | `faUpdateType` | DECODE_UPDATE |
    | sp             | 8             |

    **数据规格**

    | tensor名字 | 数据类型  | 数据格式  | 维度信息        | cpu/npu |
    | :--------- | :------- | :------- | :-------------- | ------- |
    | `lse`      | float    | nd       | [8, 16384]      | npu     |
    | `localout` | float    | nd       | [8, 16384, 128] | npu     |
    | `output`   | float    | nd       | [16384, 128]    | npu     |

