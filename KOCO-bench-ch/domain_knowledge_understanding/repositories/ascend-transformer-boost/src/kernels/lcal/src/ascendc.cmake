#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
enable_language(CCE)
# 设置编译选项
# 定义 sanitizer 相关的编译选项，如果启用则添加，否则为空
if(USE_MSSANITIZER)
    set(SANITIZER_FLAGS
        -g --cce-enable-sanitizer
    )
    set(SANITIZER_DEPEND_LIBS
        --dependent-libraries ${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-vec.a
        --dependent-libraries ${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-cube.a
    )
else()
    set(SANITIZER_FLAGS) # 空
    set(SANITIZER_DEPEND_LIBS)
endif()
set(CCE_COMPILE_OPTION
    -O2 -std=gnu++17
    --cce-aicore-only
    -Wno-deprecated-declarations
    ${SANITIZER_FLAGS}
    "SHELL:-mllvm -cce-aicore-long-call"
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm --cce-aicore-jump-expand=true"
)
set(PRIVATE_CCEC_PATH ${CMAKE_SOURCE_DIR}/3rdparty/compiler)
# 设置包含路径
if (EXISTS ${PRIVATE_CCEC_PATH})
    message(STATUS "Using custom ccec include directories")
    set(CCE_INCLUDE_BASE ${PRIVATE_CCEC_PATH})
else()
    set(CCE_INCLUDE_BASE ${ASCEND_HOME_PATH}/${ARCH}-linux)
endif()

message(STATUS "Using tikcpp include directories")
include_directories(
    ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0
    ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/interface/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/impl/
) 