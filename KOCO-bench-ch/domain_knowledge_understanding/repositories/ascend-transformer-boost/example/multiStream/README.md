# 加速库多流Demo
## 介绍
该目录下为加速库多流功能demo，其中multiStream_singleGraph_demo.cpp中为图内多流并行Demo，multiStream_multiGraph_demo.cpp中为图间同步Demo。
### 图内多流并行
multiStream_singleGraph_demo.cpp
### 图间同步
multiStream_multiGraph_demo.cpp
## 使用说明
- 首先source 对应的CANN和nnal包
    - source [cann安装路径]（默认为/usr/local/Ascend/ascend-toolkit）/set_env.sh
    - source [nnal安装路径] (默认为/usr/local/Ascend/nnal/atb) /set_env.sh
    - 如果使用加速库源码编译，source [源码路径]/output/atb目录下面的set_env.sh
- 修改当前目录下的CMakeLists.txt中的add_executable
    - 如果想要运行图内多流并行示例将其修改为：
        ```sh
        add_executable(multiStreamDemo multiStream_singleGraph_demo.cpp)
        ```
    - 如果想要运行图间同步示例将其修改为：
        ```sh
        add_executable(multiStreamDemo multiStream_multiGraph_demo.cpp)
        ```
- 生成构建系统
    - 使用cxx_abi=0
        ```sh
        mkdir build && cd build    # 创建并且进入build目录
        cmake .. -DUSE_CXX11_ABI=OFF                   # 生成构建系统
        ```
    - 使用cxx_abi=1
        ```sh
        mkdir build && cd build    # 创建并且进入build目录
        cmake .. -DUSE_CXX11_ABI=ON                   # 生成构建系统
        ```
- 编译并运行
    ```sh
    cmake --build .            # 编译项目
    ./multiStreamDemo         # 运行程序
    ```
- 查看Profiling
    ```sh
    msprof --application="multiStreamDemo"     # 生成profiling文件
    ```