#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import logging
import stat
import argparse

NAMESPACE = "Mki"

HEAD = f'''
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <functional>
#include <map>
#include "mki/utils/log/log.h"
#include "op_desc_json.h"
'''


JSON_TO_OP_FUNC = '''
{
    MKI_LOG(INFO) << opDescJson.dump(4);
    std::string opName = opDescJson["opName"];
    auto paramFunc = OP_DESC_JSON_FUNC_MAP.find(opName);
    if (paramFunc == OP_DESC_JSON_FUNC_MAP.end()) {
        return false;
    }
    try {
        paramFunc->second(opDescJson, launchParam);
    } catch(const std::exception &e) {
        MKI_LOG(ERROR) << "convert json fail, error:" << e.what();
        return false;
    }
    return true;
}'''

def parse_param_file(operation_param_file):
    struct_name = ""
    mem_list = []
    with open(operation_param_file) as fd:
        lines = fd.readlines()
        process_line = False
        end_line = False
        for org_line in lines:
            line = org_line.strip()
            line = line.strip(" ")
            fields = line.split(" ")
            if line.startswith("//"):
                continue
            if line.startswith("bool operator=="):
                end_line = True
            if end_line:
                break
            if line.startswith("struct"):
                process_line = True
            if not process_line:
                continue
            if line.startswith("enum"):
                continue
            if len(fields) == 3 and fields[0] == "struct":
                struct_name = fields[1]
            elif line.find(";") != -1 and len(fields) >= 2:
                mem_list.append([fields[0], fields[1].strip(";").strip("{0,")])
            else:
                pass
    return struct_name, mem_list


def add_default_param_code(code_lines, mem_name):
    code_lines.append(f"    try {{")
    code_lines.append(f"        param.{mem_name} = opDescJson[\"specificParam\"].at(\"{mem_name}\");")
    code_lines.append(f"    }} catch (...) {{")
    code_lines.append(f"        MKI_LOG(INFO) << \"{mem_name} not set, use default\";")
    code_lines.append(f"    }}")


def add_cast_types_code(code_lines, mem_type, mem_name):
    code_lines.append(f"    try {{")
    code_lines.append(f"        param.{mem_name} = static_cast<{mem_type}>(")
    code_lines.append(f"            opDescJson[\"specificParam\"].at(\"{mem_name}\"));")
    code_lines.append(f"    }} catch (...) {{")
    code_lines.append(f"        MKI_LOG(INFO) << \"{mem_name} not set, use default\";")
    code_lines.append(f"    }}")


def add_cast_param_code(code_lines, struct_name, mem_type, mem_name):
    code_lines.append(f"    try {{")
    code_lines.append(f"        param.{mem_name} = static_cast<{NAMESPACE}::OpParam::{struct_name}::{mem_type}>(")
    code_lines.append(f"            opDescJson[\"specificParam\"].at(\"{mem_name}\"));")
    code_lines.append(f"    }} catch (...) {{")
    code_lines.append(f"        MKI_LOG(INFO) << \"{mem_name} not set, use default\";")
    code_lines.append(f"    }}")


def get_operation_json_func(operation_param_file):
    struct_name, mem_list = parse_param_file(operation_param_file)

    code_lines = []
    code_lines.append(
        f"void {struct_name}Json(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam)")
    code_lines.append("{")
    code_lines.append(f"    {NAMESPACE}::OpParam::{struct_name} param;")
    for mem in mem_list:
        mem_type = mem[0]
        mem_name = mem[1]
        if mem_type == "bool":
            add_default_param_code(code_lines, mem_name)
        elif mem_type == "std::string":
            add_default_param_code(code_lines, mem_name)
        elif mem_type in set(("Mki::SVector<int64_t>", "SVector<int64_t>",
                "Mki::SVector<int32_t>", "SVector<int32_t>",
                "Mki::SVector<float>", "SVector<float>", "std::vector<uint32_t>")):
            type_name = mem_type.split('<')[1][:-1]
            code_lines.append("    try {")
            code_line = f"        std::vector<{type_name}> {mem_name}Values = opDescJson[\"specificParam\"].at(\"{mem_name}\");"
            code_lines.append(code_line)
            code_lines.append(
                f"        const int {mem_name}Sizes = int({mem_name}Values.size());")
            code_lines.append(
                f"        param.{mem_name}.resize({mem_name}Sizes);")
            code_lines.append(
                f"        for(int i = 0; i < {mem_name}Sizes; ++i)" + "{")
            code_lines.append(
                f"            param.{mem_name}[i] = {mem_name}Values[i];")
            code_lines.append("        }")
            code_lines.append("    } catch(...) {")
            code_lines.append(f"        MKI_LOG(INFO) << \"{mem_name} not set, use default\";")
            code_lines.append("    }")
        elif mem_type in set(("std::vector<int64_t>", "std::vector<int32_t>",
                "std::vector<float>","std::vector<int8_t>")):
            type_name = mem_type.split('<')[1][:-1]
            code_lines.append("    try {")
            code_line = f"        std::vector<{type_name}> {mem_name}Values = opDescJson[\"specificParam\"].at(\"{mem_name}\");"
            code_lines.append(code_line)
            code_lines.append(
                f"        const int {mem_name}Sizes = int({mem_name}Values.size());")
            code_lines.append(
                f"        param.{mem_name}.resize({mem_name}Sizes);")
            code_lines.append(
                f"        for(int i = 0; i < {mem_name}Sizes; ++i)" + "{")
            code_lines.append(
                f"            param.{mem_name}[i] = {mem_name}Values[i];")
            code_lines.append("        }")
            code_lines.append("    } catch(...) {")
            code_lines.append(f"        MKI_LOG(INFO) << \"{mem_name} not set, use default\";")
            code_lines.append("    }")
        elif mem_type == "float":
            add_default_param_code(code_lines, mem_name)
        elif mem_type == "double":
            add_default_param_code(code_lines, mem_name)
        elif mem_type in set(("uint32_t", "int32_t", "int64_t", "int", "size_t")):
            add_default_param_code(code_lines, mem_name)
        elif mem_type in set(("TensorDType", "Mki::TensorDType")):
            add_cast_types_code(code_lines, mem_type, mem_name)
        elif mem_type in set(("std::vector<Tensor>", "std::vector<Mki::Tensor>")):
            continue
        else:
            add_cast_param_code(code_lines, struct_name, mem_type, mem_name)
    code_lines.append("    launchParam.SetParam(param);")
    code_lines.append("}")
    return struct_name, "\n".join(code_lines)


def run(param_dir, dest_file_path):
    logging.info("param_dir: %s", param_dir)
    op_func_list = []
    for path, _, file_list in os.walk(param_dir):
        for file in file_list:
            if file == "params.h" or file == "common.h" or not file.endswith('.h'):
                continue
            op_func_list.append(
                get_operation_json_func(os.path.join(path, file)))
    if os.path.exists(dest_file_path):
        os.remove(dest_file_path)

    with os.fdopen(os.open(dest_file_path, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fd:
        fd.write(HEAD)
        fd.write(f'#include "{NAMESPACE.lower()}/ops.h"\n')
        fd.write(f'#include "{NAMESPACE.lower()}/params/params.h"\n')
        fd.write('\nnamespace Mki {\nnamespace AutoGen {\n')
        fd.write("\n")
        for _, op_func in op_func_list:
            fd.write(op_func)
            fd.write("\n\n")
        fd.write(
            "using LaunchParamSetFunc = std::function<void(const nlohmann::json &, Mki::LaunchParam &)>;\n\n")
        fd.write(
            "static const std::map<std::string, LaunchParamSetFunc> OP_DESC_JSON_FUNC_MAP = {\n")
        for op_name, _ in op_func_list:
            fd.write(
                "    {" + f"\"{op_name}Operation\", {op_name}Json" + "},\n")
        fd.write("};\n")
        fd.write("\n")
        fd.write(f"bool JsonToOpParam{NAMESPACE}(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam)\n")
        fd.write(JSON_TO_OP_FUNC)
        fd.write("\n")
        fd.write(f'''
Mki::Operation *GetOpByName{NAMESPACE}(const std::string &operationName)
{{
    return {NAMESPACE}::Ops::Instance().GetOperationByName(operationName);
}}
}}
}}''')
    logging.info("generate %s", dest_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, required=True)
    parser.add_argument('--params_path', type=str, required=True)
    input_args = parser.parse_args()
    NAMESPACE = input_args.namespace
    current_dir = os.path.split(os.path.realpath(__file__))[0]
    op_params_dir = os.path.join(current_dir, f"{input_args.params_path}/params")
    dest_file_path = os.path.join(current_dir, f"op_desc_json_{NAMESPACE.lower()}.cpp")
    run(op_params_dir, dest_file_path)
