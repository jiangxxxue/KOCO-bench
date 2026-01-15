# -*- coding: utf-8 -*-
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import configparser
import json
import logging
import shutil
import stat
import re


# sycl-target --show-targets
def get_build_target_list():
    usr_config_file_path = os.getenv("BUILD_CONFIG_FILE", '')
    if usr_config_file_path == '':
        script_file_path = os.path.realpath(__file__)
        build_config_json_file_path = os.path.join(os.path.dirname(
            script_file_path), "../src/kernels/configs/build_config.json")
    else:
        build_config_json_file_path = usr_config_file_path
    device_list = []
    try:
        with open(build_config_json_file_path) as conf_file:
            conf = json.load(conf_file)
            target_option = conf['targets']
            for target, switch in target_option.items():
                if switch is True:
                    device_list.append(target)
    except FileNotFoundError:
        logging.error("file %s is not found!", build_config_json_file_path)
        exit(1)
    except json.decoder.JSONDecodeError:
        logging.error("file %s is not json file!", build_config_json_file_path)
        exit(1)
    except KeyError:
        logging.error("key 'targets' is not found in %s!", build_config_json_file_path)
        exit(1)

    if len(device_list) == 0:
        logging.error("no target device is set")
        exit(1)

    device_list = list(set(device_list))
    return device_list


def get_info_from_file(file_path):
    result = True
    tactic_info = dict()
    magic_dict = {"RT_DEV_BINARY_MAGIC_ELF": str(0x43554245),
                  "RT_DEV_BINARY_MAGIC_ELF_AIVEC": str(0x41415246),
                  "RT_DEV_BINARY_MAGIC_ELF_AICUBE": str(0x41494343)}
    try:
        with open(file_path) as f:
            text = json.load(f)
            tactic_info["binFileName"] = text["binFileName"]
            tactic_info["compileInfo"] = text["compileInfo"]
            tactic_info["opParaSize"] = text["opParaSize"]
            tactic_info["coreType"] = text["coreType"] if text["coreType"] else ""
            magic = text["magic"]
            if magic not in magic_dict:
                logging.error("magic %s is invalid", magic)
                result = False
            else:
                tactic_info["magic"] = magic_dict[magic]
            if "kernelList" in text:
                tactic_info["kernelList"] = ','.join(
                    item["kernelName"] for item in text["kernelList"]
                )
            else:
                tactic_info["kernelList"] = text["kernelName"]
    except FileNotFoundError:
        logging.error("file %s is not found!", file_path)
        result = False
    except json.decoder.JSONDecodeError:
        logging.error("file %s is not json file!", file_path)
        result = False
    except KeyError:
        logging.error("keyerror in file %s!", file_path)
        result = False
    return tactic_info, result


def write_meta(meta_info, output_path, target_version):
    meta_path = os.path.join(output_path, 'meta.ini')
    with os.fdopen(os.open(meta_path, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w+') as fmeta:
        fmeta.write("$Version=1.0\n")
        fmeta.write(f"DeviceKernelVersion={target_version}\n")
        fmeta.write(f"$Object.Count={len(meta_info)}\n")
        for key, value in meta_info.items():
            fmeta.write(f"{key}.Object={value[0]}\n")
            fmeta.write(f"{key}.OpName={value[1]}\n")
            fmeta.write(f"{key}.KernelList={value[2]}\n")
            fmeta.write(f"{key}.CompileInfo={value[3]}\n")
            fmeta.write(f"{key}.TilingSize={value[4]}\n")
            fmeta.write(f"{key}.CoreType={value[5]}\n")
            fmeta.write(f"{key}.Magic={value[6]}\n")
        fmeta.write("$End=1\n")


# 目前只支持一个tactic文件夹下一个.o和.json文件
def copy_ascendc_code(meta_info, env_cache_dir, target_version, output_path):
    op_kernels_version_dir = os.path.join(
        env_cache_dir, "asdops_kernels", target_version)
    if not os.path.exists(op_kernels_version_dir):
        return 0
    code_file_count = 0
    for operation in os.listdir(op_kernels_version_dir):
        operation_dir = os.path.join(op_kernels_version_dir, operation)
        output_operation_dir = os.path.join(output_path, operation)
        if not os.path.exists(output_operation_dir):
            os.makedirs(output_operation_dir)
        for tactic in os.listdir(operation_dir):
            tactic_dir = os.path.join(operation_dir, tactic)
            for file in os.listdir(tactic_dir):
                if not file.endswith('.json'):
                    continue
                code_file = os.path.join(tactic_dir, "".join([file[:-4], 'o']))
                if not os.path.exists(code_file):
                    logging.error("file %s has no object file.", file)
                    exit(1)
                json_file = os.path.join(tactic_dir, file)
                tactic_info, result = get_info_from_file(json_file)
                if not result:
                    logging.error("failed to parse file %s.", json_file)
                    exit(1)
                relative_to_path = os.path.join(operation, file)
                to_path = os.path.join(output_operation_dir, file)
                shutil.copyfile(code_file, to_path)
                try:
                    compile_info_str = json.dumps(tactic_info["compileInfo"])
                    meta_info[tactic] = (
                        relative_to_path, tactic, tactic_info["kernelList"], compile_info_str,
                        tactic_info["opParaSize"], tactic_info["coreType"], tactic_info["magic"])
                except KeyError:
                    logging.error("%s get compile or meta info error", tactic)
                    exit(1)
                code_file_count += 1
    return code_file_count


def copy_tbe_code_all_version(input_paras):
    tbe_sections = input_paras["tbe_ini"].sections()
    for target_version in input_paras["target_version_list"]:
        output_path = os.path.join(
            input_paras["env_cache_dir"], "mix_obj", target_version)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        meta_info = {}
        target_version_path = os.path.join(
            input_paras["tbe_kernel_path"], target_version)

        for op_name in tbe_sections:
            op_dir_path = os.path.join(output_path, op_name)
            if not os.path.exists(op_dir_path):
                os.mkdir(op_dir_path)
            items = dict(input_paras["tbe_ini"].items(op_name))
            for op_key, relative_op_path in items.items():
                if '.' in op_key:
                    op_key, version_op_key = op_key.split('.')
                    if version_op_key != target_version:
                        continue

                tactic_info, ret = get_info_from_file(os.path.join(
                    target_version_path, relative_op_path))
                if not ret:
                    logging.error("failed to parse json file %s", relative_op_path)
                    exit(1)

                from_path = os.path.join(
                    target_version_path, "".join([relative_op_path[:-4], 'o']))
                object_name = os.path.basename(from_path)
                to_path = os.path.join(op_dir_path, object_name)
                relative_to_path = os.path.join(op_name, object_name)
                shutil.copyfile(from_path, to_path)
                if op_key not in meta_info:
                    try:
                        compile_info_str = json.dumps(tactic_info["compileInfo"])
                        meta_info[op_key] = (
                            relative_to_path, op_name, tactic_info["kernelList"], compile_info_str,
                            tactic_info["opParaSize"], tactic_info["coreType"], tactic_info["magic"])
                    except KeyError:
                        logging.error("%s get compile or meta info error", op_name)
                        exit(1)

        ascendc_file_count = copy_ascendc_code(
            meta_info, input_paras["env_cache_dir"], target_version, output_path)
        logging.info(
            f"{target_version} has {ascendc_file_count} AscendC tactics.")

        write_meta(meta_info, output_path, target_version)


def copy_tbe_device_code():
    env_code_root = os.getenv("CODE_ROOT")
    env_cache_dir = os.getenv("CACHE_DIR")
    tbe_kernel_path = os.getenv("ASCEND_KERNEL_PATH")
    if not (env_code_root and env_cache_dir and tbe_kernel_path):
        logging.error(
            "env CODE_ROOT | OUTPUT_DIR | ASDOPS_SOURCE_DIR not exist!")
        exit(1)
    logging.info(f"tbe_kernel_path: {tbe_kernel_path}")
    input_path = os.path.join(env_code_root, "configs/tbe_tactic_json.ini")
    if not os.path.exists(input_path):
        logging.error("ini file: %s not exist!", input_path)
        exit(1)
    tbe_ini = configparser.RawConfigParser()
    tbe_ini.optionxform = lambda option: option
    try:
        tbe_ini.read(input_path)
    except configparser.MissingSectionHeaderError:
        logging.error("ini file: %s format error!", input_path)
        exit(1)
    except configparser.ParsingError:
        logging.error("ini file: %s format error!", input_path)
        exit(1)

    target_version_list = get_build_target_list()
    copy_tbe_code_all_version({"env_code_root": env_code_root,
                               "target_version_list": target_version_list,
                               "env_cache_dir": env_cache_dir,
                               "tbe_kernel_path": tbe_kernel_path,
                               "tbe_ini": tbe_ini})
    os.remove(input_path)


def get_build_target_list_for_shell():
    return "\n".join(get_build_target_list())
