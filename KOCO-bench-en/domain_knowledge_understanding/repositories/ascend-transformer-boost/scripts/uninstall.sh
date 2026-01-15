#!/bin/bash
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -e
VERSION=VERSION_PLACEHOLDER
LOG_PATH=LOG_PATH_PLACEHOLDER
LOG_NAME=LOG_NAME_PLACEHOLDER
MAX_LOG_SIZE=$((1024*1024*50))
CUR_DIR=$(dirname $(readlink -f $0))

function exit_solver() {
    exit_code=$?
    if [ ${exit_code} -ne 0 ];then
        print "ERROR" "Uninstall failed, [ERROR] ret code:${exit_code}"
        exit ${exit_code}
    fi
    exit 0
}

trap exit_solver EXIT

if [ "$UID" = "0" ]; then
    log_file=${LOG_PATH}${LOG_NAME}
else
    LOG_PATH="${HOME}${LOG_PATH}"
    log_file=${LOG_PATH}${LOG_NAME}
fi
chmod 640 "${log_file}"

function print() {
    if [ ! -f "$log_file" ]; then
        if [ ! -d "${LOG_PATH}" ];then
            mkdir -p ${LOG_PATH}
        fi
        touch $log_file
    fi
    if [ x"$log_file" = x ]; then
        echo -e "[atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then 
            echo -e "[atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
            echo -e "[atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
        else
            echo -e "[atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" | tee -a $log_file
        fi
    fi
}

function delete_file_with_authority() {
    file_path=$1
    dir_path=$(dirname ${file_path})
    if [ ${dir_path} != "." ];then
        dir_authority=$(stat -c %a ${dir_path})
        chmod 700 "${dir_path}"
        if [ -d ${file_path} ];then
            rm -rf "${file_path}"
        else
            rm -f "${file_path}"
        fi
        chmod ${dir_authority} "${dir_path}"
    else
        chmod 700 "${file_path}"
        if [ -d ${file_path} ];then
            rm -rf "${file_path}"
        else
            rm -f "${file_path}"
        fi
    fi
}

function delete_installed_files() {
    install_dir=$1
    csv_path=$install_dir/scripts/filelist.csv
    is_first_line=true
    cd "$install_dir"
    if [ ! -f $csv_path ];then
        print "INFO" "filelist.csv is not founded, uninstall by delete whole folder."
        [ -n "$1" ] && rm -rf "$1"
        return 0
    fi
    cat ${csv_path} | while read line
    do
        if [ ${is_first_line} == "false" ];then
            file_path=$(echo ${line} | awk '{print $1}')
            if [ ! -f ${file_path} ];then
                continue
            fi
            delete_file_with_authority $file_path
        fi
        is_first_line=false
    done
}

function delete_latest() {
    cd "$1/.."
    if [ -d "latest" -a $(readlink -f $1/../latest) == $1 ];then
        rm -f latest
    fi

    if [ -f "set_env.sh" ];then
        chmod 700 set_env.sh
        rm -f set_env.sh
    fi
}

function delete_empty_recursion() {
    if [ ! -d $1 ];then
        return 0
    fi
    for file in $1/*
    do
        if [ -d $file ];then
            delete_empty_recursion $file
        fi
    done
    if [ -z "$(ls -A $1)" ];then
        delete_file_with_authority $1
    fi
}

function uninstall_process() {
    #检查对应版本目录下的文件是否需要删除，是则进行删除
    if [ ! -d $1 ];then
        print "ERROR" "Ascend-cann-atb dir of $VERSION is not exist, uninstall failed!"
        return 0
    fi
    print "INFO" "Ascend-cann-atb uninstall $(basename $1) start!"
    atb_dir=$(cd "$1/..";pwd)
    delete_latest $1
    delete_installed_files $1
    if [ -d $1 ];then
        delete_empty_recursion $1
    fi
    if [ -z "$(ls $atb_dir)" ];then
        rm -rf "$atb_dir"
    fi
    print "INFO" "Ascend-cann-atb uninstall $(basename $1) success!"
}

function uninstall_torch_atb() {
    PACKAGE_NAME="torch_atb"
    if pip show $PACKAGE_NAME > /dev/null 2>&1; then
        print "INFO" "torch_atb is installed. need uninstall."
        pip uninstall -y torch_atb
    fi
}

install_path=$(cd "${CUR_DIR}/../../${VERSION}";pwd)
uninstall_process ${install_path}
uninstall_torch_atb
chmod 440 "${log_file}"