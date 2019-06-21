#ifndef XFUSION_DEVICE_MALLOC_H
#define XFUSION_DEVICE_MALLOC_H

#include <xfusion/macros.h>

namespace fusion
{

FUSION_HOST void *deviceMalloc(size_t size);
FUSION_HOST void deviceRelease(void **dev_ptr);

} // namespace fusion

#endif