#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""This procedure is used to average the three performance outcomes and compare with the baseline.

This programme is mainly used for automated testing; the automated testing programme will store the single performance result in a testcase.
If it is used for manual testing, save the single performance result to a specified directory or modify the input parameters of the programme.
The input parameter is the name of the operation to be calculated.
"""

import pandas as pd
import sys
import os
import glob
import argparse

class CompareWithBaseline:
    def __init__(self, kernelName, baseResult, matched_files):
        self.kernelName = kernelName
        self.baseResult = baseResult
        self.matched_files = matched_files
        print("matched_files is", self.matched_files)

    """ Find the op_summary*.csv of the Ms_prof results. """
    def get_matched_files_df(self):
        if self.matched_files:
            df = pd.read_csv(self.matched_files[0])
        else:
            print("No /PROF_*/mindstudio_profiler_output/op_summary*.csv exiting, please check!")
            sys.exit(1)
        return df

    def calculate_averange_time(self):
        """ Calculate the mean of the three performance results. """
        #result_single_for_cal = self.read_three_times_result()
        result_df = self.get_matched_files_df()
        mla_dataSet = result_df[result_df["Op Name"] == kernelName]
        kernel_data = mla_dataSet.tail(80)
        kernel_average_duration = kernel_data["Task Duration(us)"].mean()
        print("Task Duration(us).mean() is", kernel_data["Task Duration(us)"].mean())

        if not kernel_average_duration:
            print(f'result time of {self.kernelName} had something abnormal,please check the op_summary*.csv.')
            sys.exit(1)

        return kernel_average_duration

    def compare_with_baseline(self):
        kernel_average_duration = self.calculate_averange_time()
        self.baseResult = float(self.baseResult)
        print("kernel_average_duration is", kernel_average_duration)
        print("self.baseResult is", self.baseResult)
        if kernel_average_duration > self.baseResult * 1.05:
            print(f"{self.kernelName}", ":This version's performance is worse than baseline.")
            print("Failed")
        else:
            print(f"{self.kernelName}", ":This version's performance is equal with baseline.")
            print("Pass")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate 80 Times average time of KernelOp and compare with baseline")
    parser.add_argument("-i", type=str, default="MLAPreprocessKernel", help='input the testing OperationKernalName')
    parser.add_argument("-b", type=str, default="1000", help='input the baseResult')
    args = parser.parse_args()
    kernelName = args.i
    baseResult = args.b
    print("kernelName is", kernelName)
    print("baseResult is", baseResult)

    curDir = os.getcwd()
    print("Current Directory is", curDir)
    pattern = f"{curDir}/PROF_*/mindstudio_profiler_output/op_summary*.csv"
    matched_files = glob.glob(pattern)
    for file in matched_files:
        print("matched_files is", matched_files)

    cal_and_com = CompareWithBaseline(kernelName, baseResult, matched_files)
    cal_and_com.compare_with_baseline()
