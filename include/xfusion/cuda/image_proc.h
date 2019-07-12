#ifndef FUSION_CUDA_IMGPROC_H
#define FUSION_CUDA_IMGPROC_H

#include <opencv2/core/cuda.hpp>

namespace fusion
{

void create_inverse_depth_map(const cv::cuda::GpuMat &depth, cv::cuda::GpuMat &inv_depth);

}

#endif