#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "memory_env.h"
class MemoryPool {
public:
 explicit MemoryPool(size_t poolSize);
 ~MemoryPool();
 void AllocateBlock(uint32_t size, int &blockId);
 void FreeBlock(int blockId);
 void GetBlockPtr(int blockId, void *&addr);
private:
 uint64_t GenerateBlocksId();
 std::atomic<uint64_t> id_ = 0;
 std::mutex blockMutex_;
 void *baseMemPtr_ = nullptr;
 void *curMemPtr_ = nullptr;
 int64_t remainSize_ = 0;
 std::unordered_map<int, MemoryBlock> freeBlocks_;
 std::unordered_map<int, MemoryBlock> usedBlocks_;
};
#endif