# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import argparse
import subprocess
import os
import random
import stat
import logging
import signal

PACK_DIR = ""
PACK_NAME = ""
SHELL_TEMP_FILE_LIST = []


def init_globals():
    """
    初始化全局变量 PACK_DIR 和 PACK_NAME。

    这个函数首先尝试从环境变量 ATB_PACK_PATH 获取路径，然后根据架构名称生成 PACK_DIR。
    如果找不到路径或运行包，将返回 False。

    返回:
        bool: 如果成功初始化则返回 True，否则返回 False。
    """
    global PACK_DIR, PACK_NAME
    atb_pack_path = os.getenv("ATB_PACK_PATH")
    if atb_pack_path is None:
        logging.error("ATB_PACK_PATH is not set in environment variables.")
        return False

    arch = subprocess.check_output(["arch"]).decode("utf-8").strip()
    PACK_DIR = os.path.join(atb_pack_path, arch)
    logging.info("Generated PACK_DIR: %s", PACK_DIR)
    if not os.path.isdir(PACK_DIR):
        logging.error("Pack dir is not exist")
        return False

    files_in_dir = os.listdir(PACK_DIR)
    if len(files_in_dir) > 0:
        PACK_NAME = files_in_dir[0]
        logging.info("First file found in PACK_DIR: %s", PACK_NAME)
    else:
        logging.error("Run pack is not exist")
        return False

    return True


def generate_random_hex_string():
    """
    生成一个随机的十六进制字符串。

    这个函数用于生成唯一的 shell 脚本文件名。

    返回:
        str: 一个随机的十六进制字符串。
    """
    hex_digits = "0123456789abcdef"
    return "".join(random.choice(hex_digits) for _ in range(8))


def generate_tmp_shell_path(absolute_path):
    """
    生成一个临时 shell 脚本文件名。
    """
    return os.path.join(
        os.path.dirname(absolute_path),
        f"{os.path.splitext(os.path.basename(absolute_path))[0]}_{generate_random_hex_string()}.sh",
    )


def generate_shell_path(absolute_path):
    """
    生成一个唯一的 shell 脚本文件路径。

    这个函数在给定的绝对路径下生成一个唯一的 shell 脚本文件路径，直到找到一个不存在的文件名为止。

    返回:
        str: 生成的 shell 脚本文件路径。
    """
    shell_path = absolute_path
    while os.path.exists(shell_path):
        shell_path = generate_tmp_shell_path(absolute_path)
    return shell_path


def replace_placeholders_with_globals(commands):
    """
    用全局变量替换命令中的占位符。

    这个函数将命令中的 "SCPTS_PACK_DIR" 和 "SCPTS_PACK_NAME" 占位符替换为全局变量 PACK_DIR 和 PACK_NAME 的值。

    返回:
        list: 替换占位符后的命令列表。
    """
    global PACK_DIR, PACK_NAME
    fixed_commands = []
    for command in commands:
        command = command.replace("SCPTS_PACK_DIR", PACK_DIR)
        command = command.replace("SCPTS_PACK_NAME", PACK_NAME)
        fixed_commands.append(command)
    return fixed_commands


def get_commands_codes(absolute_path):
    """
    从给定的绝对路径读取文件，并将文件中的命令和代码分别提取出来。

    参数:
    absolute_path (str): 文件的绝对路径。

    返回:
    commands (list): 提取出的命令列表。
    codes (list): 提取出的代码列表。

    异常:
    如果命令和代码的数量不匹配，则打印错误信息并返回None, None。
    """
    with open(absolute_path, "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    commands = []
    codes = []
    for i, line in enumerate(lines, start=0):
        if i % 2 == 0:
            commands.append(line)
        else:
            codes.append(line)
    if len(commands) != len(codes):
        logging.error("Error: commands and codes are not match")
        return None, None
    commands = replace_placeholders_with_globals(commands)
    codes = [int(code) for code in codes]
    return commands, codes


def check_code(actualcodes, expectcodes):
    """
    检查实际代码和期望代码是否匹配。

    参数:
    actualcodes: 实际代码的列表
    expectcodes: 期望代码的列表

    返回值:
    results: 包含每行代码比较结果的列表
    status: 如果所有代码都匹配，则为True，否则为False
    """
    results = []
    status = True
    for i, (actualcode, expectcode) in enumerate(zip(actualcodes, expectcodes)):
        linecase = f"Line [{i*2+1}] actual value: {actualcode}, expect value: {expectcode}"
        if actualcode != expectcode:
            status = False
            results.append(f"{linecase}, Conclusion: fail")
        else:
            results.append(f"{linecase}, Conclusion: success")
    return results, status


def handle_input(args):
    """
    处理输入的函数，根据输入的模式调用不同的处理函数。

    参数:
    args: 包含输入路径和模式的对象。

    返回值:
    如果输入路径存在，返回处理函数的状态；否则返回False。

    异常描述:
    如果输入路径不存在，打印"No Input Path"。
    如果没有输入路径，打印"No Input"。
    """
    # 获取当前工作目录
    current_path = os.getcwd()
    # 将相对于工作目录的输入路径转换为绝对路径
    absolute_path = os.path.join(current_path, args.input)
    if os.path.exists(absolute_path):
        if args.mode == "execute":
            logging.info("Executing file: %s with mode: execute", args.input)
            return handle_input_execute(absolute_path)
        else:
            logging.info("Executing file: %s with mode: check", args.input)
            return handle_input_check(absolute_path)

    logging.error("No Input Path: %s", args.input)
    return False


def handle_input_check(absolute_path):
    """
    处理输入的绝对路径，执行其中的命令并返回执行结果。

    参数:
    absolute_path (str): 需要处理的绝对路径。

    返回:
    status (int): 执行结果状态码。
    """
    global SHELL_TEMP_FILE_LIST
    commands, codes = get_commands_codes(absolute_path)

    shell = generate_shell_path(absolute_path)
    SHELL_TEMP_FILE_LIST.append(shell)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(shell, flags, modes), "w") as fout:
        fout.write("#!/bin/bash\nfunction write_code() {\n    echo $1\n    return $1\n}\n")
        for command in commands:
            fout.write(f"{command} > /dev/null 2>&1\n")
            fout.write(f"write_code $?\n")
    process = subprocess.run(
        ["bash", shell],
        shell=False,
        cwd=os.path.dirname(absolute_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.remove(shell)
    resultcodes = process.stdout.splitlines()
    resultcodes = [int(code) for code in resultcodes]
    results, status = check_code(resultcodes, codes)
    for result in results:
        print(f"Case [{os.path.basename(absolute_path)}] {result}")
    return status


def handle_input_execute(absolute_path):
    """
    处理输入的绝对路径，生成一个shell脚本，并执行该脚本。

    参数:
    absolute_path (str): 输入的绝对路径。

    返回:
    status (bool): 执行脚本后的状态，如果执行成功则返回True，否则返回False。
    """
    global SHELL_TEMP_FILE_LIST
    commands, _ = get_commands_codes(absolute_path)

    shell = generate_shell_path(absolute_path)
    SHELL_TEMP_FILE_LIST.append(shell)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(shell, flags, modes), "w") as fout:
        fout.write("#!/bin/bash\n")
        for command in commands:
            fout.write(f"{command} && ")
        fout.write(f'echo "{absolute_path}" execute done\n')
    process = subprocess.run(
        ["bash", shell],
        shell=False,
        cwd=os.path.dirname(absolute_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.remove(shell)
    results = process.stdout.splitlines()
    for result in results:
        print(f"{result}")
    status = process.returncode == 0
    return status


def handle_dir(args):
    """
    处理目录的函数，遍历目录下的所有文件，如果是文件则判断是否为.txt文件，如果是则添加到txts_list列表中，
    如果是目录则递归调用handle_dir函数处理，最后将所有.txt文件的路径传入handle_input函数处理。

    参数:
    args: 包含目录路径的参数对象

    返回值:
    status: 处理结果，如果所有文件都处理成功则返回True，否则返回False
    """

    # 获取当前工作目录
    current_path = os.getcwd()
    # 将相对于工作目录的目录路径转换为绝对路径
    abs_dir_path = os.path.join(current_path, args.dir)
    if os.path.exists(abs_dir_path):
        status = True
        files_list = os.listdir(abs_dir_path)
        files_list = sorted(files_list)
        txts_list = []
        for file_name in files_list:
            file_path = os.path.join(abs_dir_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith(".txt"):
                txts_list.append(file_path)
            if os.path.isdir(file_path):
                temp_args = args
                temp_args.dir = file_path
                status = handle_dir(temp_args) and status
        temp_args = args
        for txt in txts_list:
            temp_args.input = txt
            status = handle_input(temp_args) and status
        return status
    logging.error("No Dir Path: %s", args.dir)
    return False


def clean_up_temp_files():
    """
    清理临时文件的函数。

    这个函数会遍历SHELL_TEMP_FILE_LIST列表中的所有文件，如果文件存在，则删除该文件，并记录日志。

    参数:
    无

    返回值:
    无

    异常描述:
    如果文件删除失败，会抛出异常。
    """
    global SHELL_TEMP_FILE_LIST

    # 使用列表推导式和os.remove来删除所有临时文件
    for file in SHELL_TEMP_FILE_LIST:
        if os.path.exists(file):
            os.remove(file)
            logging.info("File %s has been removed.", file)


def signal_handler(signum, frame):
    """
    信号处理函数，当接收到信号时，执行清理临时文件的操作，并记录日志信息。

    参数:
    signum: 接收到的信号编号
    frame: 当前的堆栈帧

    返回值:
    无

    异常描述:
    如果清理临时文件失败，会抛出SystemExit异常，并返回状态码
    """
    # 使用懒插值记录日志
    logging.info("Received signal %d, performing cleanup.", signum)
    clean_up_temp_files()
    logging.info("Cleanup completed.")
    # 使用懒插值记录退出信息
    logging.info("Exiting with status %d", 128 + signum)
    raise SystemExit(128 + signum)


def create_arg_parser():
    """
    创建命令行参数解析器，用于解析用户输入的命令行参数。

    参数:
    无

    返回值:
    parser: 返回一个解析器对象，用于解析命令行参数。

    异常描述:
    无
    """
    parser = argparse.ArgumentParser(
        "This is a script designed to automatically execute scripts and verify their return codes."
    )
    parser.add_argument(
        "--mode",
        default="check",
        help="Specify the operation mode. 'execute' for execute, 'check' or other values for debug.",
    )
    parser.add_argument(
        "--input", help="Path to the input file. This file will be processed according to the specified mode."
    )
    parser.add_argument(
        "--dir",
        help="Path to the directory containing the input files. The script will process all files in this directory.",
    )
    return parser


def parse_args(parser):
    args = parser.parse_args()
    return args


def main():
    """主函数，负责处理程序的主要流程。

    功能描述：
    1. 注册信号处理器，捕获 SIGINT 信号。
    2. 创建参数解析器，解析参数。
    3. 初始化全局变量，处理输入文件或目录。
    4. 根据处理结果，打印相应的日志信息并退出程序。

    参数：
    无

    异常描述：
    如果初始化全局变量失败，或者输入文件或目录参数不合法，会打印错误日志并退出程序。
    """
    # 注册信号处理器，捕获 SIGINT 信号
    signal.signal(signal.SIGINT, signal_handler)

    # 创建参数解析器
    parser = create_arg_parser()

    # 解析参数
    args = parse_args(parser)

    status = init_globals()
    if not status:
        logging.error("Initialization of global variables failed")
        exit(1)

    status = args.input or args.dir
    if not status:
        logging.error("Either input file or input dir must be provided.")
        exit(1)

    status = args.input and args.dir
    if status:
        logging.error("Cannot have both input and dir parameters specified.")
        exit(1)

    # 根据 input 参数处理文件
    if args.input:
        status = handle_input(args)
        if status:
            logging.info("Input file processed successfully.")
            exit(0)
        else:
            logging.error("Error processing.")
            exit(1)

    # 根据 dir 参数处理目录
    if args.dir:
        status = handle_dir(args)
        if status:
            logging.info("Directory processed successfully.")
            exit(0)
        else:
            logging.error("Error processing.")
            exit(1)


if __name__ == "__main__":
    main()
