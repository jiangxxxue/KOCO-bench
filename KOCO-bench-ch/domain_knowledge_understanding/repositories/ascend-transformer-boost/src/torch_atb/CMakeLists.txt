#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

file(GLOB_RECURSE pybind11_source_files "*.cpp")
pybind11_add_module(_C ${pybind11_source_files})
set_target_properties(_C PROPERTIES OUTPUT_NAME "_C" SUFFIX ".so")
target_link_options(_C PRIVATE -Wno-odr -rdynamic -ldl -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none -fexceptions)
target_link_libraries(_C PRIVATE torch_npu)
target_include_directories(_C PRIVATE ${ATB_INCLUDE_DIR})
install(TARGETS _C DESTINATION ${CMAKE_SOURCE_DIR}/output/torch_atb)