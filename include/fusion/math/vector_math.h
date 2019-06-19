#ifndef FUSION_MATH_TYPES_H
#define FUSION_MATH_TYPES_H

#include <cmath>
#include <macros.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sophus/se3.hpp>

using uchar = unsigned char;

template <class T>
struct Vector2
{
    FUSION_HOST_AND_DEVICE inline Vector2() : x(0), y(0) {}
    FUSION_HOST_AND_DEVICE inline Vector2(const Vector2<T> &other) : x(other.x), y(other.y) {}
    FUSION_HOST_AND_DEVICE inline Vector2(T x, T y) : x(x), y(y) {}
    FUSION_HOST_AND_DEVICE inline Vector2(T val) : x(val), y(val) {}

    FUSION_HOST_AND_DEVICE inline Vector2<T> operator+(const Vector2<T> &other) const
    {
        return Vector2<T>(x + other.x, y + other.y);
    }

    FUSION_HOST_AND_DEVICE inline Vector2<T> operator-(const Vector2<T> &other) const
    {
        return Vector2<T>(x - other.x, y - other.y);
    }

    FUSION_HOST_AND_DEVICE inline T operator*(const Vector2<T> &other) const
    {
        return x * other.x + y * other.y;
    }

    FUSION_HOST_AND_DEVICE inline Vector2<T> operator/(const T val) const
    {
        return Vector2<T>(x / val, y / val);
    }

    T x, y;
};

template <class T>
struct Vector3
{
    FUSION_HOST_AND_DEVICE inline Vector3() : x(0), y(0), z(0) {}
    FUSION_HOST_AND_DEVICE inline Vector3(const Vector3<T> &other) : x(other.x), y(other.y), z(other.z) {}
    FUSION_HOST_AND_DEVICE inline Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    FUSION_HOST_AND_DEVICE inline Vector3(T val) : x(val), y(val), z(val) {}

    FUSION_HOST_AND_DEVICE inline bool operator==(const Vector3<T> &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator+=(const Vector3<T> &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator+(const Vector3<T> &other) const
    {
        return Vector3<T>(x + other.x, y + other.y, z + other.z);
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator-(const Vector3<T> &other) const
    {
        return Vector3<T>(x - other.x, y - other.y, z - other.z);
    }

    FUSION_HOST_AND_DEVICE inline T operator*(const Vector3<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator/(const T val) const
    {
        return Vector3<T>(x / val, y / val, z / val);
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator%(const T val) const
    {
        return Vector3<T>(x % val, y % val, z % val);
    }

    FUSION_HOST_AND_DEVICE inline float norm() const
    {
        return sqrt((float)(x * x + y * y + z * z));
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> cross(const Vector3<T> &other) const
    {
        return Vector3<T>(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    T x, y, z;
};

template <class T>
struct Vector4
{
    FUSION_HOST_AND_DEVICE inline Vector4() : x(0), y(0), z(0), w(0) {}
    FUSION_HOST_AND_DEVICE inline Vector4(const Vector3<T> &other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
    FUSION_HOST_AND_DEVICE inline Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    FUSION_HOST_AND_DEVICE inline Vector4(T val) : x(val), y(val), z(val), w(val) {}
    FUSION_HOST_AND_DEVICE inline Vector4(const Vector3<T> &other, const T &val) : x(other.x), y(other.y), z(other.z), w(val) {}

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator+(const Vector4<T> &other) const
    {
        return Vector4<T>(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator-(const Vector4<T> &other) const
    {
        return Vector4<T>(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    FUSION_HOST_AND_DEVICE inline T operator*(const Vector4<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator/(const T val) const
    {
        return Vector4<T>(x / val, y / val, z / val, w / val);
    }

    T x, y, z, w;
};

using Vector2i = Vector2<int>;
using Vector2s = Vector2<short>;
using Vector2f = Vector2<float>;
using Vector2c = Vector2<uchar>;
using Vector2d = Vector2<double>;

using Vector3i = Vector3<int>;
using Vector3f = Vector3<float>;
using Vector3c = Vector3<uchar>;
using Vector3d = Vector3<double>;

using Vector4i = Vector4<int>;
using Vector4f = Vector4<float>;
using Vector4c = Vector4<uchar>;
using Vector4d = Vector4<double>;

template <class T, class U>
FUSION_HOST_AND_DEVICE inline Vector3<T> operator*(T lhs, Vector3<U> rhs)
{
    return Vector3<T>(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs);
}

FUSION_HOST_AND_DEVICE inline Vector3f operator+(Vector3i lhs, Vector3f rhs)
{
    return Vector3f(rhs.x + lhs.x, rhs.y + lhs.y, rhs.z + lhs.z);
}

FUSION_HOST_AND_DEVICE inline Vector3c ToVector3c(const Vector3f &other)
{
    return Vector3c((int)other.x, (int)other.y, (int)other.z);
}

FUSION_HOST_AND_DEVICE inline Vector3i ToVector3i(const Vector3f &V)
{
    Vector3i b((int)V.x, (int)V.y, (int)V.z);
    b.x = b.x > V.x ? b.x - 1 : b.x;
    b.y = b.y > V.y ? b.y - 1 : b.y;
    b.z = b.z > V.z ? b.z - 1 : b.z;
    return b;
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3<T> ToVector3(const Vector4<T> &other)
{
    return Vector3<T>(other.x, other.y, other.z);
}

FUSION_HOST_AND_DEVICE inline Vector3f normalised(const Vector3f &V)
{
    return V / V.norm();
}

FUSION_HOST_AND_DEVICE inline Vector3i operator*(const Vector3i lhs, int rhs)
{
    return Vector3i(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3f operator*(const Vector3<T> &lhs, float rhs)
{
    return Vector3f(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3f ToVector3f(const Vector3<T> &V)
{
    return Vector3f((float)V.x, (float)V.y, (float)V.z);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3<T> floor(const Vector3<T> &V)
{
    return Vector3<T>(floor(V.x), floor(V.y), floor(V.z));
}

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

#endif
