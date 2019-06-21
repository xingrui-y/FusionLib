#ifndef FUSION_ICP_ROBUST_ESTIMATE_H
#define FUSION_ICP_ROBUST_ESTIMATE_H

#include <sophus/se3.hpp>
#include <xfusion/core/intrinsic_matrix.h>
#include <opencv2/cudaarithm.hpp>

namespace fusion
{

void rgb_step(
    const cv::cuda::GpuMat &curr_intensity,
    const cv::cuda::GpuMat &last_intensity,
    const cv::cuda::GpuMat &last_vmap,
    const cv::cuda::GpuMat &curr_vmap,
    const cv::cuda::GpuMat &intensity_dx,
    const cv::cuda::GpuMat &intensity_dy,
    cv::cuda::GpuMat &sum,
    cv::cuda::GpuMat &out,
    const float stddev_estimate,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix K,
    float *jtj, float *jtr,
    float *residual);

} // namespace fusion

#endif