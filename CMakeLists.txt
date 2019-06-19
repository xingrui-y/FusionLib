cmake_minimum_required(VERSION 3.1)
project(FusionLIB VERSION 0.1 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA 10.1 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

add_library(fusion STATIC "")

target_sources(fusion
PRIVATE
    # core modules
    src/fusion/core/cuda_imgproc.cu
    src/fusion/core/intrinsic_matrix.cpp

    # icp
    src/fusion/icp/pose_estimator.cu
    src/fusion/icp/robust_estimator.cu

    # voxel mapping
    src/fusion/mapping/depth_fusion.cu
    src/fusion/mapping/map_struct.cu
    src/fusion/mapping/ray_tracing.cu
    src/fusion/mapping/scene_mesh.cu
)

target_include_directories(fusion
PUBLIC
    include/    
    include/fusion/
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
PRIVATE
    src/fusion/core/
    src/fusion/icp/
    src/fusion/mapping/
    include/fusion/core/
    include/fusion/icp/
    include/fusion/math/
    include/fusion/mapping/
)

set_target_properties(fusion
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 11
)

target_link_directories(fusion
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)