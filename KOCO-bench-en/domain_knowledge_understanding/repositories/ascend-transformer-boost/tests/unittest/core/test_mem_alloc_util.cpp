/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "atb/utils/mem_allocation_solver/mem_allocation_solver_creator.h"
#include <gtest/gtest.h>

using namespace atb;

TEST(TESTMEMALLOC, TESTBFSOLVER)
{
    setenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "0", 1);
    BruteforceMemAllocationSolver* memAllocationSolver = new BruteforceMemAllocationSolver();
    ASSERT_NE(memAllocationSolver, nullptr);
    void* addr0 = memAllocationSolver->GetOffset(6);
    void* addr1 = memAllocationSolver->GetOffset(5);
    void* addr2 = memAllocationSolver->GetOffset(4);
    void* addr3 = memAllocationSolver->GetOffset(3);
    void* addr4 = memAllocationSolver->GetOffset(2);
    void* addr5 = memAllocationSolver->GetOffset(1);
    memAllocationSolver->Free(addr0);
    memAllocationSolver->Free(addr1);
    memAllocationSolver->Free(addr2);
    memAllocationSolver->Free(addr3);
    memAllocationSolver->Free(addr4);
    void* addr6 = memAllocationSolver->GetOffset(5);
    void* addr7 = memAllocationSolver->GetOffset(15);
    EXPECT_EQ(memAllocationSolver->GetSize(), 41);
    memAllocationSolver->Reset();
    delete memAllocationSolver;
}

TEST(TESTMEMALLOC, TESTBLOCKSOLVER)
{
    setenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "1", 1);
    BlockMemAllocationSolver* memAllocationSolver = new BlockMemAllocationSolver();
    ASSERT_NE(memAllocationSolver, nullptr);
    void* addr0 = memAllocationSolver->GetOffset(6);
    void* addr1 = memAllocationSolver->GetOffset(5);
    void* addr2 = memAllocationSolver->GetOffset(4);
    void* addr3 = memAllocationSolver->GetOffset(3);
    void* addr4 = memAllocationSolver->GetOffset(2);
    void* addr5 = memAllocationSolver->GetOffset(1);
    memAllocationSolver->Free(addr0);
    memAllocationSolver->Free(addr1);
    memAllocationSolver->Free(addr2);
    memAllocationSolver->Free(addr3);
    memAllocationSolver->Free(addr4);
    void* addr6 = memAllocationSolver->GetOffset(5);
    void* addr7 = memAllocationSolver->GetOffset(15);
    EXPECT_EQ(memAllocationSolver->GetSize(), 36);
    memAllocationSolver->Reset();
    delete memAllocationSolver;
}

TEST(TESTMEMALLOC, TESTHEAPSOLVER)
{
    setenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "2", 1);
    HeapMemAllocationSolver* memAllocationSolver = new HeapMemAllocationSolver();
    ASSERT_NE(memAllocationSolver, nullptr);
    void* addr0 = memAllocationSolver->GetOffset(6);
    void* addr1 = memAllocationSolver->GetOffset(5);
    void* addr2 = memAllocationSolver->GetOffset(4);
    void* addr3 = memAllocationSolver->GetOffset(3);
    void* addr4 = memAllocationSolver->GetOffset(2);
    void* addr5 = memAllocationSolver->GetOffset(1);
    memAllocationSolver->Free(addr0);
    memAllocationSolver->Free(addr1);
    memAllocationSolver->Free(addr2);
    memAllocationSolver->Free(addr3);
    memAllocationSolver->Free(addr4);
    memAllocationSolver->Free(addr5);
    void* addr6 = memAllocationSolver->GetOffset(5);
    void* addr7 = memAllocationSolver->GetOffset(15);
    EXPECT_EQ(memAllocationSolver->GetSize(), 21);
    memAllocationSolver->Reset();
    delete memAllocationSolver;
}

TEST(TESTMEMALLOC, TESTNOBLOCKSOLVER)
{
    setenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "3", 1);
    NoblockMemAllocationSolver* memAllocationSolver = new NoblockMemAllocationSolver();
    ASSERT_NE(memAllocationSolver, nullptr);
    void* addr0 = memAllocationSolver->GetOffset(6);
    void* addr1 = memAllocationSolver->GetOffset(5);
    void* addr2 = memAllocationSolver->GetOffset(4);
    void* addr3 = memAllocationSolver->GetOffset(3);
    void* addr4 = memAllocationSolver->GetOffset(2);
    void* addr5 = memAllocationSolver->GetOffset(1);
    memAllocationSolver->Free(addr0);
    memAllocationSolver->Free(addr1);
    memAllocationSolver->Free(addr2);
    memAllocationSolver->Free(addr3);
    memAllocationSolver->Free(addr4);
    void* addr6 = memAllocationSolver->GetOffset(5);
    void* addr7 = memAllocationSolver->GetOffset(15);
    EXPECT_EQ(memAllocationSolver->GetSize(), 21);
    memAllocationSolver->Reset();
    delete memAllocationSolver;
}