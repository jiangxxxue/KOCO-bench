#include <acl/acl.h>
#include "memory_utils.h"
#include "utils/log.h"
#include "utils/utils.h"
static MemoryManager g_memoryManager;
MemoryManager::MemoryManager() {}
void MemoryManager::CreateMemoryPool(size_t poolSize)
{
 uint32_t deviceCount = 0;
 CHECK_RET(aclrtGetDeviceCount(&deviceCount), "get devicecount fail");
 for (size_t i = 0; i < deviceCount; i++) {
 aclrtSetDevice(i);
 std::shared_ptr<MemoryPool> memoryPool = std::make_shared<MemoryPool>(poolSize);
 memoryPools_.push_back(memoryPool);
 LOG_INFO("create mempool for device " + std::to_string(i) + " success");
 }
}
int32_t MemoryManager::GetDeviceId()
{
 int32_t deviceId = -1;
 CHECK_RET(aclrtGetDevice(&deviceId), "get device ID fail");
 return deviceId;
}
std::shared_ptr<MemoryPool> &MemoryManager::GetMemoryPool()
{
 size_t deviceId = static_cast<size_t>(GetDeviceId());
  CHECK_RET(deviceId >= memoryPools_.size(), "Invalid device id " + deviceId);
 return memoryPools_[deviceId];
}
void MemoryManager::AllocateBlock(uint32_t size, int &blockId)
{
 GetMemoryPool()->AllocateBlock(size, blockId);
}
void MemoryManager::FreeBlock(int blockId)
{
 GetMemoryPool()->FreeBlock(blockId);
}
void MemoryManager::GetBlockPtr(int blockId, void *&addr)
{
 GetMemoryPool()->GetBlockPtr(blockId, addr);
}
MemoryManager &GetMemoryManager()
{
 return g_memoryManager;
}