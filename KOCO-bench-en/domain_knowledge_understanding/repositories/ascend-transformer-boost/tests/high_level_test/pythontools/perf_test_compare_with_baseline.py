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

This programme is mainly used for automated testing; the automated testing programme will store the single performance result in a specified directory.
If it is used for manual testing, save the single performance result to a specified directory or modify the input parameters of the programme.
The input parameter is the name of the operation to be calculated.
"""

import csv
import argparse
import sys
from os.path import dirname, abspath

class CompareWithBaseline:
    def __init__(self, operation_name, result_times_file_name):
        self.op_name = operation_name
        self.times = result_times_file_name
        self.work_path = dirname(dirname(abspath(__file__))) + f'/{self.op_name}/Performance'

    def read_three_times_result(self):
        """ Return single-time results from the specified result.csv."""
        result_single = [[]]

        for time in self.times:
            result_single_path = self.work_path + f'/PerformanceResult/{time}'
            result_single_csv = result_single_path + f'/{self.op_name}_Perf_TestCase_csvopstest_result.csv'
            testcase_num = 0
            with open(result_single_csv, 'r') as file:
                result_single_csv_data = csv.reader(file, delimiter='|')
                for row in result_single_csv_data:
                    if row.count('TotalTime(us)') != 0:
                        index_time = row.index('TotalTime(us)')
                        continue
                    testcase_num += 1
                    if len(result_single) < testcase_num:
                        result_single.append([])
                    result_single[testcase_num - 1].append(float(row[index_time]))

        return result_single

    def calculate_averange_time(self):
        """ Calculate the mean of the three performance results. """
        result_single_for_cal = self.read_three_times_result()
        if not result_single_for_cal[0]:
            print(f'result time of {self.op_name} had something abnormal,please check the result.csv.')
            sys.exit(1)

        result_average = []
        for test_case in result_single_for_cal:
            time_sum = sum(test_case)
            result_average.append(time_sum / len(self.times))

        return result_average

    def compare_with_baseline(self):
        """ Compare with baseline using mean result of specified op. """
        result_average = self.calculate_averange_time()
        if not result_average:
            print(f'average time of {self.op_name} had something abnormal,please check the result.csv.')
            sys.exit(1)

        result_baseline_path = self.work_path
        result_baseline_csv = result_baseline_path + f'/{self.op_name}_BaseLine.csv'
        with open(result_baseline_csv, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            baseline_info = []
            for row in reader:
                baseline_info.append(row)

        baseline_data = []
        for line in baseline_info:
            if line.count('BaseLine(us)') != 0:
                index_time = line.index('BaseLine(us)')
                continue
            baseline_data.append(line[index_time])

        for i, base in enumerate(baseline_data):
            if float(base) > result_average[i]:
                print(f"{self.op_name}" + " TestCase", i, ":This version's performance is better than baseline.")
                baseline_info[i + 1][2] = result_average[i]
            else:
                if float(base) * 1.05 > result_average[i]:
                    print(f"{self.op_name}" + " TestCase", i, ":This version's performance is equal with baseline.")
                else:
                    print(f"{self.op_name}" + " TestCase", i, ":This version's performance is worse than baseline.")
                    level = abs(float(base) - result_average[i]) / float(base) * 100
                    print("The degradation of this version:", round(level, 3), "%")

        testcase_num = 0
        head = True
        for row_list in baseline_info:
            if head == True:
                row_list.append('TotalTime_new(us)')
                head = False
                continue
            row_list.append(str(result_average[testcase_num]))
            testcase_num += 1

        with open(result_baseline_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter="|")
            for row in baseline_info:
                writer.writerow(row)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate average time and compare with baseline")
    parser.add_argument("-i", type=str, default="ElewiseOperation", help='input the testing OperationName')
    args = parser.parse_args()
    OperationName = args.i

    cal_and_com = CompareWithBaseline(OperationName, ['first','second','third'])
    cal_and_com.compare_with_baseline()