#include <xfusion/core/device_malloc.h>
#include <xfusion/core/cuda_utils.h>
#include "xfusion/core/cuda_safe_call.h"

namespace fusion
{

FUSION_HOST void *deviceMalloc(size_t sizeByte)
{
    void *dev_ptr;
    safe_call(cudaMalloc((void **)&dev_ptr, sizeByte));
    return dev_ptr;
}

FUSION_HOST void deviceRelease(void **dev_ptr)
{
    if (*dev_ptr != NULL)
        safe_call(cudaFree(*dev_ptr));

    *dev_ptr = 0;
}

} // namespace fusion
