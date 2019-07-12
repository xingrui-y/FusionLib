#ifndef FUSION_UTILS_CUDA_UTILS_H
#define FUSION_UTILS_CUDA_UTILS_H

#include <iostream>
#include <cuda_runtime.h>

#define MAX_THREAD 1024
#define MAX_WARP_SIZE 32

template <typename NumType1, typename NumType2>
static inline int div_up(NumType1 dividend, NumType2 divisor)
{
    return (int)((dividend + divisor - 1) / divisor);
}

template <class FunctorType>
__global__ void call_device_functor(FunctorType device_functor)
{
    device_functor();
}

#endif