#ifndef FUSION_MAPPING_DATA_TYPES_H
#define FUSION_MAPPING_DATA_TYPES_H

#include <macros.h>
#include <vector_math.h>
#include <cuda_runtime_api.h>

struct FUSION_EXPORT Voxel
{
    FUSION_DEVICE inline float get_sdf() const;
    FUSION_DEVICE inline void set_sdf(float val);
    FUSION_DEVICE inline float get_weight() const;
    FUSION_DEVICE inline void set_weight(float val);

    short sdf;
    float weight;
    uchar3 rgb;
};

FUSION_DEVICE inline float unpack_float(short val)
{
    return val / (float)32767;
}

FUSION_DEVICE inline short pack_float(float val)
{
    return (short)(val * 32767);
}

FUSION_DEVICE inline float Voxel::get_sdf() const
{
    return unpack_float(sdf);
}

FUSION_DEVICE inline void Voxel::set_sdf(float val)
{
    sdf = pack_float(val);
}

FUSION_DEVICE inline float Voxel::get_weight() const
{
    return weight;
}

FUSION_DEVICE inline void Voxel::set_weight(float val)
{
    weight = val;
}

#endif