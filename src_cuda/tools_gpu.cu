#include <cstdint>
#include <cstdio>
#include <nvml.h>
#include "tools_gpu.h"

#if defined(RTE_RRTMGP_GPU_MEMPOOL_CUDA)
static bool cuda_mempool_initialized = false;

void prepare_cuda_mempool()
{
    if (cuda_mempool_initialized)
        return;

    printf("Setting up CUDA mempool.\n");
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    auto threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    cuda_mempool_initialized = true;
}
#endif

namespace Tools_gpu {
unsigned long long energy_usage_gpu() {
    static bool is_initialized = false;
    nvmlReturn_t code;

    if (!is_initialized) {
        code = nvmlInit();

        if (code != NVML_SUCCESS) {
            fprintf(stderr, "warning: nvmlInit failed: %s\n", nvmlErrorString(code));
            return 0;
        }

        is_initialized = true;
    }

    int device_index;
    cuda_safe_call(cudaGetDevice(&device_index));

    char pci_bus_id[64];
    cuda_safe_call(cudaDeviceGetPCIBusId(pci_bus_id, 64, device_index));

    nvmlDevice_t device;
    code = nvmlDeviceGetHandleByPciBusId(pci_bus_id, &device);
    if (code != NVML_SUCCESS) {
        fprintf(stderr, "warning: nvmlDeviceGetHandleByPciBusId failed: %s\n", nvmlErrorString(code));
        return 0;
    }

    unsigned long long energy_mj;
    code = nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);

    if (code != NVML_SUCCESS) {
        fprintf(stderr, "warning: nvmlDeviceGetTotalEnergyConsumption failed: %s\n", nvmlErrorString(code));
        return 0;
    }

    return energy_mj;
}
}
