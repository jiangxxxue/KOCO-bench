#ifndef MEMORY_ENV_H
#define MEMORY_ENV_H
#include <iostream>
struct MemoryBlock {
 int64_t blockId;
 size_t blockSize;
 void *address = nullptr;
};
#endif