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
install_flag=n
uninstall_flag=n
upgrade_flag=n
recover_flag=n
install_path_flag=n
install_for_all_flag=n
torch_atb_flag=n
install_dir=""
install_torch_atb_dir=""
default_install_path="/usr/local/Ascend/atb"
target_dir=""
version_dir=""
sourcedir=$PWD
VERSION=VERSION_PLACEHOLDER
LOG_PATH=LOG_PATH_PLACEHOLDER
LOG_NAME=LOG_NAME_PLACEHOLDER
MAX_LOG_SIZE=$((1024*1024*50))

if [ "$UID" = "0" ]; then
    log_file=${LOG_PATH}${LOG_NAME}
    install_for_all_flag=y
else
    LOG_PATH="${HOME}${LOG_PATH}"
    log_file=${LOG_PATH}${LOG_NAME}
fi

function exit_solver() {
    exit_code=$?
    if [ ${exit_code} -ne 0 ];then
        print "ERROR" "Command execution failed, [ERROR] ret code:${exit_code}"
        if [[ "${upgrade_flag}" == "y" && "${recover_flag}" == "y" ]]; then
            recover_old_version
        fi
        exit ${exit_code}
    fi
    exit 0
}

trap exit_solver EXIT

# 将日志记录到日志文件
function log() {
    if [ ! -f "$log_file" ]; then
        if [ ! -d "${LOG_PATH}" ];then
            mkdir -p ${LOG_PATH}
        fi
        touch $log_file
    fi
    if [ x"$log_file" = x ]; then
        echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then 
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
        else
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" >>$log_file
        fi
    fi
}

# 将日志记录到日志文件并打屏
function print() {
    if [ ! -f "$log_file" ]; then
        if [ ! -d "${LOG_PATH}" ];then
            mkdir -p ${LOG_PATH}
        fi
        touch $log_file
    fi
    if [ x"$log_file" = x ]; then
        echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then 
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
        else
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" | tee -a $log_file
        fi
    fi
}

# 创建文件夹
function make_dir() {
    log "INFO" "mkdir ${1}"
    mkdir -p ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 failed !"
        exit 1
    fi
}

# 创建文件
function make_file() {
    log "INFO" "touch ${1}"
    touch ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 failed !"
        exit 1
    fi
}

## 日志模块初始化 ##
function log_init() {
    # 判断输入的日志保存路径是否存在，不存在就创建
    if [ ! -d "$LOG_PATH" ]; then
        make_dir "$LOG_PATH"
    fi
    chmod 750 "${LOG_PATH}"
    # 判断日志文件是否存在，如果不存在就创建；存在则判断是否大于50M
    if [ ! -f "$log_file" ]; then
        make_file "$log_file"
        # 安装日志权限
        chmod_recursion ${LOG_PATH} "750" "dir"
        chmod 640 "${log_file}"
    else
        local filesize=$(ls -l $log_file | awk '{ print $5}')
        local maxsize=$((1024*1024*50))
        if [ $filesize -gt $maxsize ]; then
            local log_file_move_name="ascend_atb_install_bak.log"
            mv -f "${log_file}" "${LOG_PATH}${log_file_move_name}"
            chmod 440 "${LOG_PATH}${log_file_move_name}"
            make_file "$log_file"
            log "INFO" "log file > 50M, move ${log_file} to ${LOG_PATH}${log_file_move_name}."
        fi
        chmod 640 "${log_file}"
    fi
    print "INFO" "Install log save in ${log_file}"
}

function chmod_authority() {
    # 修改文件和目录权限
    chmod_file ${default_install_path}
    chmod_file ${install_dir}
    chmod_file ${install_torch_atb_dir}
    chmod 440 "${install_dir}/scripts/filelist.csv"
    local file_rights=$([ "${install_for_all_flag}" == "y" ] && echo 555 || echo 550)
    chmod ${file_rights} "${install_dir}/scripts/uninstall.sh"
    chmod ${file_rights} "${install_dir}/install.sh"
    chmod_dir ${default_install_path} "550"
    chmod_dir ${install_dir} "550"
    local path_rights=$([ "${install_for_all_flag}" == "y" ] && echo 755 || echo 750)
    chmod ${path_rights} "${default_install_path}"
    chmod ${path_rights} "${install_dir}"
    if [ -n "$install_torch_atb_dir" ] && [ -d "$install_torch_atb_dir" ]; then
        chmod ${path_rights} "$install_torch_atb_dir"
    fi
}

function chmod_file() { 
    chmod_recursion ${1} "550" "file" "*.sh"
    chmod_recursion ${1} "440" "file" "*.bin"
    chmod_recursion ${1} "440" "file" "*.h"
    chmod_recursion ${1} "440" "file" "*.info"
    chmod_recursion ${1} "440" "file" "*.so"
    chmod_recursion ${1} "440" "file" "*.ini"
    chmod_recursion ${1} "440" "file" "*.a"
}

function chmod_dir() {
    chmod_recursion ${1} ${2} "dir"
} 

function chmod_recursion() {
    # install-for-all 实际上是给other组用户赋予了和同组用户相同的权限
    local parameter2=$2
    local rights="$(echo ${parameter2:0:2})""$(echo ${parameter2:1:1})"
    rights=$([ "${install_for_all_flag}" == "y" ] && echo ${rights} || echo $2)
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -name "$4" -exec chmod ${rights} {} \; 2>/dev/null
    fi
}

function parse_script_args() {
    while true
    do
        case "$1" in
        --quiet)
            QUIET="y"
            shift
        ;;
        --install)
        install_flag=y
        shift
        ;;
        --install-path=*)
        install_path_flag=y
        target_dir=$(echo $1 | cut -d"=" -f2-)
        target_dir=${target_dir}/atb
        shift
        ;;
        --uninstall)
        uninstall_flag=y
        shift
        ;;
        --install-for-all)
        install_for_all_flag=y
        shift
        ;;
        --upgrade)
        upgrade_flag=y
        shift
        ;;
        --torch_atb)  
        torch_atb_flag=y
        shift
        ;;
        --*)
        shift
        ;;
        *)
        break
        ;;
        esac
    done
}

function check_target_dir_owner() {
    local cur_owner=$(whoami)
    if [ "$cur_owner" != "root" ];then
        return
    fi
    #计算被'/'符号分割的段数
    local seg_num=$(expr $(echo ${1} | grep -o "/" | wc -l) + "1")
    local path=""
    #根据段数遍历所有路径
    for((i=1;i<=$seg_num;i++))
    do
        local split=$(echo ${1} | cut -d "/" -f$i)
        if [ "$split" = "" ];then
            continue
        fi
        local path=${path}"/"${split}
        if [ -d "${path}" ]; then
            local path_owner=$(stat -c %U "${path}")
            if [ "$path_owner" != "root" ]; then
                print "ERROR" "Install failed, install path or its parents path owner [$path_owner] is inconsistent with current user [$cur_owner]."
                exit 1
            fi
        fi
    done
}

function check_path() {
    if [ ! -d "${install_dir}" ]; then
        mkdir -p ${install_dir}
        if [ ! -d "${install_dir}" ]; then
            print "ERROR" "Install failed, [ERROR] create ${install_dir} failed"
            exit 1
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
            delete_file_with_authority ${file_path}
        fi
        is_first_line=false
    done

    # 主动移除过去版本遗漏文件
    residual_files=(
        "atb/cxx_abi_0/configs/platform_configs/Ascend910_9361.ini"
        "atb/cxx_abi_0/configs/platform_configs/Ascend910_9382.ini"
        "atb/cxx_abi_0/configs/platform_configs/Ascend910_9392.ini"
        "atb/cxx_abi_1/configs/platform_configs/Ascend910_9361.ini"
        "atb/cxx_abi_1/configs/platform_configs/Ascend910_9382.ini"
        "atb/cxx_abi_1/configs/platform_configs/Ascend910_9392.ini"
    )
    for file_path in ${residual_files[@]}; do
        if [ -f ${file_path} ];then
            delete_file_with_authority ${file_path}
        fi
    done
}

function delete_latest() {
    cd "${default_install_path}"
    if [ -d "latest" ];then
        rm -f latest
    fi

    if [ -f "set_env.sh" ];then
        chmod 700 set_env.sh
        rm -f set_env.sh
    fi
}

function check_csv_exist() {
    #判断是否存在csv文件，存在则按文件删除，不存在则整文件夹删除
    install_dir=$1
    csv_path=$install_dir/scripts/filelist.csv
    if [ -f $csv_path ];then
        return 0
    else
        return 1
    fi
}

function uninstall_process() {
    #检查对应版本目录下的文件是否需要删除，是则进行删除
    if [ ! -d $1 ];then
        return 0
    fi
    print "INFO" "Ascend-cann-atb $(basename $1) uninstall start!"
    atb_dir=$(cd "$1/..";pwd)
    delete_latest $1
    delete_installed_files $1
    uninstall_torch_atb
    if [ -d $1 ];then
        delete_empty_recursion $1
    fi
    if [ "$2" == "y" -a -z "$(ls $atb_dir)" ];then
        rm -rf "$atb_dir"
    fi
    print "INFO" "Ascend-cann-atb $(basename $1) uninstall success!"
}

function install_to_path() {
    install_dir=${default_install_path}/${VERSION}
    if [ -d ${install_dir} ];then
        print "INFO" "The installation directory exists, uninstall first.."
    fi
    uninstall_process ${install_dir}
    check_target_dir_owner ${install_dir}
    check_path
    cd "${install_dir}"
    install_torch_atb
    copy_files
    [ -f "${default_install_path}/set_env.sh" ] && rm -rf "${default_install_path}/set_env.sh"
    mv "${install_dir}/set_env.sh" "${default_install_path}"
    cd "${default_install_path}"
    ln -snf $VERSION latest
}

function copy_files() {
    cp -r "${sourcedir}/atb" "$install_dir"
    cp -r "${sourcedir}/scripts" "$install_dir"
    cp -r "${sourcedir}/whl" "$install_dir"
    cp "${sourcedir}/install.sh" "$install_dir"
    cp "${sourcedir}/version.info" "$install_dir"
    cp "${sourcedir}/set_env.sh" "$install_dir"
}

function install_process() {
    local arch_pkg=ATBPKGARCH
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        ARCH="aarch64"
    fi
    if [ -n "${ARCH}" ]; then
        if [ "${arch_pkg}" != "${ARCH}" ]; then
            print "ERROR" "Install failed, pkg arch ${arch_pkg} is not consistent with the current enviroment architecture ${ARCH}."
            exit 1
        fi
    fi
    if [ -n "${target_dir}" ]; then
        if [[ ! "${target_dir}" = /* ]]; then
            print "ERROR" "Install failed, [ERROR] use absolute path for --install-path argument"
            exit 1
        fi
        install_to_path
    else
        install_to_path
    fi
}

function check_owner() {
    local cur_owner=$(whoami)

    if [ "${ASCEND_TOOLKIT_HOME}" == "" -a "${ASCEND_NNAE_HOME}" == "" ];then
        print "ERROR" "Check owner failed, please check env ASCEND_TOOLKIT_HOME or ASCEND_NNAE_HOME is set."
        exit 1
    fi

    if [ "${ASCEND_HOME_PATH}" == "" ]; then
        print "ERROR" "Check owner failed, please check env ASCEND_HOME_PATH is set."
        exit 1
    else
        cann_path=${ASCEND_HOME_PATH}
    fi

    if [ ! -d "${cann_path}" ]; then
        print "ERROR" "Check owner failed, can not find cann in ${cann_path}."
        exit 1
    fi
    cann_owner=$(stat -c %U "${cann_path}")

    if [ "${cann_owner}" != "${cur_owner}" ]; then
        print "ERROR" "Check owner failed, current owner is not same with CANN."
        exit 1
    fi

    if [[ "${cur_owner}" != "root" && "${install_flag}" == "y" ]]; then
        default_install_path="${HOME}/Ascend/atb"
    fi

    if [ "${install_path_flag}" == "y" ]; then
        default_install_path="${target_dir}"
    fi

    print "INFO" "Check owner success!"
}

function uninstall() {
    install_dir=${default_install_path}/${VERSION}
    uninstall_process ${install_dir} y
}

function check_uninstall_path() {
    [ -f "$log_file" ] && chmod 640 "${log_file}"
    local cur_owner=$(whoami)
    if [ "${install_path_flag}" == "y" ]; then
        default_install_path="${target_dir}"
    else
        if [ "${cur_owner}" != "root" ]; then
            default_install_path="${HOME}/Ascend/atb"
        fi
    fi
    
    if [ ! -d "${default_install_path}" ]; then
        print "ERROR" "Uninstall failed, can not find the path of Ascend-cann-atb."
        exit 1
    fi
}

function check_upgrade_path() {
    local cur_owner=$(whoami)
    if [ "${install_path_flag}" == "y" ]; then
        default_install_path="${target_dir}"
    else
        if [ "${cur_owner}" != "root" ]; then
            default_install_path="${HOME}/Ascend/atb"
        fi
    fi

    if [ ! -d "${default_install_path}" ]; then
        print "ERROR" "Upgrade failed, can not find the path of Ascend-cann-atb."
        exit 1
    else
        if [ ! -d "${default_install_path}/latest" -o ! -f "${default_install_path}/set_env.sh" ]; then
            print "ERROR" "Can not find file set_env.sh or latest!"
            exit 1
        fi
        version_dir=$(readlink -f "${default_install_path}/latest")
        local path_owner=$(stat -c %U "${version_dir}")
        if [ "${path_owner}" != "${cur_owner}" ]; then
            print "ERROR" "Upgrade failed, the current owner is not same as CANN's."
            exit 1
        fi
    fi

    print "INFO" "Check upgrade path success!"
}

function back_up_old_version() {
    back_up_dir="${version_dir}_recover"
    if [ -d "${back_up_dir}" ]; then
        print "ERROR" "Upgrade failed, *_recover file is exist, can not back up the old version of Ascend-cann-atb."
        exit 1
    fi

    cp -rp "${version_dir}" "${back_up_dir}"
    cp -p "${default_install_path}/set_env.sh" "${default_install_path}/set_env_recover.sh"
    print "INFO" "back up the old Ascend-cann-atb version success!"
}

function recover_old_version() {
    if [ -d "${version_dir}" ]; then
        chmod -R 700 "${version_dir}"
        rm -rf "${version_dir}"
    fi
    if [ -f "${default_install_path}/set_env.sh" ]; then
        chmod 700 "${default_install_path}/set_env.sh"
        rm -f "${default_install_path}/set_env.sh"
    fi
    mv "${back_up_dir}" "${version_dir}"
    mv "${default_install_path}/set_env_recover.sh" "${default_install_path}/set_env.sh"
    local version=$(basename ${version_dir})
    cd "${default_install_path}"
    ln -snf ${version} latest
    print "INFO" "recover old Ascend-cann-atb version success!"
}

function remove_back_up_version() {
    chmod -R 700 "${back_up_dir}"
    rm -r "${back_up_dir}"
    chmod 700 "${default_install_path}/set_env_recover.sh"
    rm -f "${default_install_path}/set_env_recover.sh"
    print "INFO" "finish remove back up version!"
}

function upgrade() {
    torch_atb_flag=$1
    # 先备份旧版本，再卸载旧版本，安装新版本，卸载备份版本
    [ -f "$log_file" ] && chmod 640 "${log_file}"
    print "INFO" "Ascend-cann-atb uninstall start!"
    check_owner
    check_upgrade_path
    back_up_old_version
    if pip show "torch_atb" > /dev/null 2>&1; then
        print "INFO" "torch_atb is installed. need upgrade."
        torch_atb_flag=y
    fi
    uninstall_process ${version_dir}
    recover_flag=y
    install_process
    print "INFO" "finish install process!"
    chmod_authority
    remove_back_up_version
    print "INFO" "Ascend-cann-atb upgrade success!"
}

function install_torch_atb() {
    if [ "${torch_atb_flag}" != "y" ]; then
        return 0
    fi

    py_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    py_major_version=${py_version%%.*}
    py_minor_version=${py_version##*.}

    if [ "$py_major_version" != "3" ] || { [ "$py_minor_version" != "10" ] && [ "$py_minor_version" != "11" ]; }; then
        echo "ERROR: Unsupported Python version. Only Python 3.10, and 3.11 are supported."
        exit 1
    fi

    wheel_file="torch_atb-0.0.1-cp${py_major_version}${py_minor_version}-none-any.whl"
    wheel_path="${sourcedir}/whl/${wheel_file}"

    if [ ! -f "$wheel_path" ]; then
        echo "ERROR: Wheel file ${wheel_file} not found at ${wheel_path}."
        exit 1
    fi

    if pip show torch_atb > /dev/null 2>&1; then
        echo "INFO: torch_atb is already installed, force-reinstalling..."
        if pip install --force-reinstall "$wheel_path" > /dev/null 2>&1; then
            echo "INFO: torch_atb reinstallation succeeded!"
        else
            echo "ERROR: torch_atb reinstallation failed!"
            exit 1
        fi
    else
        echo "INFO: torch_atb not found, installing..."
        if pip install "$wheel_path" > /dev/null 2>&1; then
            echo "INFO: torch_atb installation succeeded!"
        else
            echo "ERROR: torch_atb installation failed!"
            exit 1
        fi
    fi
}

function uninstall_torch_atb() {
    if pip show "torch_atb" > /dev/null 2>&1; then
        print "INFO" "torch_atb is installed. need uninstall."
        pip uninstall -y torch_atb
    fi
}

function main() {
    parse_script_args $*
    if [ "${uninstall_flag}" == "y" ]; then
        check_uninstall_path
        print "INFO" "Ascend-cann-atb uninstall start!"
        uninstall
    elif [ "${upgrade_flag}" == "y" ]; then
        upgrade "${torch_atb_flag}"
    elif [[ "${install_path_flag}" == "y" || "${install_flag}" == "y" ]]; then
        log_init
        check_owner
        install_process
        chmod_authority
        print "INFO" "Ascend-cann-atb install success!"
    fi
    chmod 440 "${log_file}"
}

main $*
