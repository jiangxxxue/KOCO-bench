#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import sys
import os
import argparse
import re
import argparse

def FilterCommentsStringMacro(codeLine):
    regMacro = r'\s*#.*$'
    regComments = r'/\*.*?\*/|//.*$'
    regString1 = r'\\"'
    regString2 = r'"[^"]*"'
    codeLine = re.sub(regMacro, "", codeLine)
    codeLine = re.sub(regComments, "", codeLine)
    codeLine = re.sub(regString1, "", codeLine)
    codeLine = re.sub(regString2, "", codeLine)
    return codeLine

def FilterTemplates(codeLine):
    regPrefix = r'\b(const|volatile|typename)\b'
    codeLine = re.sub(regPrefix, "", codeLine)
    regTemp = r'<\s*(::)?\w+\s*(::\s*\w+\s*)*(,\s*(::)?\w+\s*(::\s*\w+\s*)*)*([*&]\s*)?>'
    i = 0
    while(i<50):
        currentLine = codeLine
        codeLine = re.sub(regTemp, "", codeLine)
        if codeLine == currentLine:
            break
        i+=1
    regSpecial = r'(->|>>|<<|::)'
    codeLine = re.sub(regSpecial, "", codeLine)
    return codeLine

def WhetherConditionals(codeLine):
    regulation = r'([?|!~><]|&&|==|!=|\b(if|switch|case|while|for)\b)'
    newCodeLine = re.sub(regulation, "", codeLine)
    if len(newCodeLine) < len(codeLine):
        return True
    else:
        return False

class BranchFilter:
    def __init__(self, debug):
        self.iter_ = 0
        self.limit_ = 5
        self.macroList = []
        self.debugSwitch = int(debug)
        self.dirDirtyWords = ['/llt/', '/ai/', '/vector/',
                              '/x86/', '/sve/dev/operators/', 
                              '/test/', '/adapter/']

    def DEBUG_LOG(self, args):
        if self.debugSwitch:
            print('[DEBUG] ', args)

    def LoadFile(self, cppFile):
        with open(cppFile) as fw:
            fileLineList = fw.readlines()
            return fileLineList

    def FindAllMacros(self, codeLines):
        for line in codeLines:
            line = line[0:-1]
            regMacro = r'\s*#define\s*'
            newLine = re.sub(regMacro, "", line)
            if len(newLine) < len(line):
                regSpace = r'\s*\\'
                newLine = re.sub(regSpace, "", newLine)
                regBraces = r'\(.*\)'
                newLine = re.sub(regBraces, "", newLine)
                self.DEBUG_LOG(newLine)
                self.macroList.append(newLine)

    def FindAllCodeFiles(self, path):
        for root,dirs,files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)

                escapeFlag = False
                for key in self.dirDirtyWords:
                    if key in file_path:
                        # self.DEBUG_LOG('bye', file_path)
                        escapeFlag = True
                if escapeFlag:
                    continue

                splitFile = file.split('.')
                if len(splitFile) < 1:
                    continue

                fix = splitFile[len(splitFile) - 1]
                if fix == 'cpp' or fix == 'h' or fix == 'hpp':
                    self.DEBUG_LOG(file_path)
                    codeLines = self.LoadFile(file_path)
                    self.FindAllMacros(codeLines)


    def WhetherCounterNewFile(self, line):
        prefix = line[0:3]
        if prefix == 'SF:':
            return True
        else:
            return False

    def GetBranchInfo(self, currentLine):
        branchInfo = currentLine.split(':')[1]
        branchInfo = branchInfo.split(',')
        cppLineNumber = int(branchInfo[0])
        return cppLineNumber


    def GetCppDistanceBetweenBranches(self, i, infoLines):
        currentLine = infoLines[i][0:-1]
        currentCppLineNumber = self.GetBranchInfo(currentLine)
        cppLineNumber = currentCppLineNumber
        while cppLineNumber == currentCppLineNumber:
            i += 1
            cppLineNumber = self.GetBranchInfo(infoLines[i][0:-1])
        limit = cppLineNumber - currentCppLineNumber
        return abs(limit)


    def CheckMacrosInRef(self, line):
        flag = False
        for key in self.macroList:
            if key in line:
                flag = True
        return flag

    def ProcessOneLine(self, infoLines, cppLines, newInfoData):
        currentLine = infoLines[self.iter_][0:-1]
        currentCppLineNumber = self.GetBranchInfo(currentLine)
        cppLineNumber = currentCppLineNumber
        self.DEBUG_LOG(cppLineNumber)
        '''
        Here to process the cpp file to determine whether this piece of code would
        be remained.
        1. Filter the comment/string/macros/templates
        2. check the macros
        3. get the keywords like(if, &&, ....), flag set to true
        '''
        limit = min(self.limit_, self.GetCppDistanceBetweenBranches(self.iter_, infoLines))
        self.DEBUG_LOG("limit" + str(self.GetCppDistanceBetweenBranches(self.iter_, infoLines)))
        startLine = cppLineNumber - 1
        endLine = min(startLine + limit, len(cppLines)-1)
        Flag = False
        for iterLine in range(startLine, endLine):
            cppLine = cppLines[iterLine][0:-1] ## cpp counts from 1
            self.DEBUG_LOG(cppLine)
            cppLine = FilterCommentsStringMacro(cppLine)
            cppLine = FilterTemplates(cppLine)
            self.DEBUG_LOG(cppLine)
            Flag = WhetherConditionals(cppLine) | self.CheckMacrosInRef(cppLine)
            self.DEBUG_LOG(Flag)
            if Flag:
                break
        '''
        Once the process is done
        Conditions above decides whether this infoLine should be push_back
        '''
        while cppLineNumber == currentCppLineNumber:
            self.DEBUG_LOG(infoLines[self.iter_][0:-1])
            # push_back or not
            if Flag:
                newInfoData += infoLines[self.iter_]
            self.iter_ += 1
            cppLineNumber = self.GetBranchInfo(infoLines[self.iter_][0:-1])
            
        return newInfoData

    def ProcessOneFile(self, infoLines, line, newInfoData):
        '''
        load the cpp file and delete some branches in info file
        '''
        fileAbsPath = line[3:-1]
        cppLines = self.LoadFile(fileAbsPath)

        deletedNum = 0
        while infoLines[self.iter_][0:13] != 'end_of_record':
            prefix = infoLines[self.iter_][0:4]
            if prefix == 'BRDA': # prefix 'BRF:' or prefix 'BRH:' is not useful!
                newInfoData = self.ProcessOneLine(infoLines, cppLines, newInfoData)
            else:
                # not BRDA, just push back
                newInfoData += infoLines[self.iter_]
                self.iter_ += 1
        return newInfoData

    def MainLoopInfo(self, infoLines, newInfoFile):
        newInfoData = ""

        while self.iter_ < len(infoLines):
            line = infoLines[self.iter_]
            if self.WhetherCounterNewFile(line):
                self.DEBUG_LOG('[INFO] Process File in:' + line[3:-1])
                newInfoData = self.ProcessOneFile(infoLines, line, newInfoData)
            else:
                self.iter_ += 1
                newInfoData += line

        with open(newInfoFile, 'w') as fw:
            fw.write(newInfoData)


    def Filter(self, inputFile, outFile, rootDir):
        self.FindAllCodeFiles(rootDir)
        infoLines = self.LoadFile(inputFile)
        self.MainLoopInfo(infoLines, outFile)

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='11')
    parser.add_argument('--output', help='11')
    parser.add_argument('--root', help='11')
    parser.add_argument('--debug', help='11')
    args = parser.parse_args()

    inputFile = args.input
    outFile = args.output
    rootDir = args.root
    debug = args.debug

    bf = BranchFilter(debug)
    bf.Filter(inputFile, outFile, rootDir)
    print("[INFO] DONE")
    #'/home/zz/workspace/ads_develop/iads/common/operator/sve/dev/operators_online/Core/Ops/resize/')

main()