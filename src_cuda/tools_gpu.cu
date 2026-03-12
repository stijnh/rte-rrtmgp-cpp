#if defined(RTE_RRTMGP_GPU_MEMPOOL_CUDA)
#include <cstdint>
#include <cstdio>
#include <kmm/kmm.hpp>

using kmm::GPUmemoryPool;

static bool cuda_mempool_initialized = false;

void prepare_cuda_mempool()
{
    if (cuda_mempool_initialized)
        return;

    printf("Setting up CUDA mempool.\n");
    GPUmemoryPool mempool;
    gpuDeviceGetDefaultMemPool(&mempool, 0);
    auto threshold = UINT64_MAX;
    gpuMemPoolSetAttribute(mempool, gpuMemPoolAttrReleaseThreshold, &threshold);
    cuda_mempool_initialized = true;
}
#endif
