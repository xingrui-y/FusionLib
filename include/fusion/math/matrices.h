#ifndef FUSION_MATH_ROTATION_H
#define FUSION_MATH_ROTATION_H

#include <fusion/macros.h>
#include <fusion/math/vectors.h>

namespace fusion
{

class DeviceMatrix3x4
{
public:
    DeviceMatrix3x4() = default;
    DeviceMatrix3x4(const Sophus::SE3d &pose)
    {
        Eigen::Matrix<float, 4, 4> mat = pose.cast<float>().matrix();
        row_0 = Vector4f(mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3));
        row_1 = Vector4f(mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3));
        row_2 = Vector4f(mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3));
    }

    __host__ __device__ Vector3f rotate(const Vector3f &pt) const
    {
        Vector3f result;
        result.x = row_0.x * pt.x + row_0.y * pt.y + row_0.z * pt.z;
        result.y = row_1.x * pt.x + row_1.y * pt.y + row_1.z * pt.z;
        result.z = row_2.x * pt.x + row_2.y * pt.y + row_2.z * pt.z;
        return result;
    }

    __host__ __device__ Vector3f operator()(const Vector3f &pt) const
    {
        Vector3f result;
        result.x = row_0.x * pt.x + row_0.y * pt.y + row_0.z * pt.z + row_0.w;
        result.y = row_1.x * pt.x + row_1.y * pt.y + row_1.z * pt.z + row_1.w;
        result.z = row_2.x * pt.x + row_2.y * pt.y + row_2.z * pt.z + row_2.w;
        return result;
    }

    __host__ __device__ Vector4f operator()(const Vector4f &pt) const
    {
        Vector4f result;
        result.x = row_0.x * pt.x + row_0.y * pt.y + row_0.z * pt.z + row_0.w;
        result.y = row_1.x * pt.x + row_1.y * pt.y + row_1.z * pt.z + row_1.w;
        result.z = row_2.x * pt.x + row_2.y * pt.y + row_2.z * pt.z + row_2.w;
        result.w = 1.0f;
        return result;
    }

    Vector4f row_0, row_1, row_2;
};

} // namespace fusion

#endif