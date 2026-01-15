#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H
#include <memory>
#include <vector>
#include "memorypool.h"
class MemoryManager {
public:
 MemoryManager();
 void CreateMemoryPool(size_t poolSize);
 int32_t GetDeviceId();
 std::shared_ptr<MemoryPool> &GetMemoryPool();
 void AllocateBlock(uint32_t size, int &blockId);
 void FreeBlock(int blockId);
 void GetBlockPtr(int blockId, void *&addr);
private:
 std::vector<std::shared_ptr<MemoryPool>> memoryPools_;
};
MemoryManager &GetMemoryManager();
#endif